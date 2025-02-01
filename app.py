import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import folium
from flask import Flask, render_template
import math
from collections import defaultdict
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Constants
TRAVEL_TIME_PER_KM = 5  # minutes
DELIVERY_TIME_PER_SHIPMENT = 10  # minutes

def haversine(coord1, coord2):
    """Calculate Haversine distance between two coordinates in kilometers"""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371  # Earth radius in km
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def load_data():
    """Load and preprocess data"""
    shipments = pd.read_excel('SmartRoute Optimizer.xlsx', sheet_name='Shipments_Data')
    vehicles = pd.read_excel('SmartRoute Optimizer.xlsx', sheet_name='Vehicle_Information')
    store = pd.read_excel('SmartRoute Optimizer.xlsx', sheet_name='Store Location').iloc[0]
    
    # Fix for FutureWarning - explicit conversion
    vehicles['Number'] = vehicles['Number'].map(lambda x: 9999 if x == 'Any' else int(x))
    vehicles['Max Trip Radius (in KM)'] = vehicles['Max Trip Radius (in KM)'].map(
        lambda x: 1e9 if x == 'Any' else float(x)
    )
    
    return shipments, vehicles, (store['Latitute'], store['Longitude'])

def cluster_shipments(shipments, store_coords, vehicles_df):
    """Enhanced clustering with capacity-aware splitting"""
    clustered = []
    max_capacity = min(25, vehicles_df['Shipments_Capacity'].max())
    
    for timeslot, group in shipments.groupby('Delivery Timeslot'):
        if group.empty: continue
            
        coords = group[['Latitude', 'Longitude']].values
        n_clusters = max(len(group) // max_capacity, 1)
        
        # Use KMeans directly instead of DBSCAN
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(coords)
        
        group = group.copy()
        group['cluster'] = labels
        
        # Process each cluster
        processed_clusters = []
        for cluster_id in group['cluster'].unique():
            cluster_df = group[group['cluster'] == cluster_id].copy()  # Create explicit copy
            
            # Sort by distance to store
            distances = cluster_df.apply(
                lambda row: haversine(store_coords, (row['Latitude'], row['Longitude'])),
                axis=1
            )
            cluster_df['distance_to_store'] = distances
            cluster_df = cluster_df.sort_values('distance_to_store')
            
            # Split if needed
            while len(cluster_df) > max_capacity:
                processed_clusters.append(cluster_df.iloc[:max_capacity])
                cluster_df = cluster_df.iloc[max_capacity:]
            
            if not cluster_df.empty:
                processed_clusters.append(cluster_df)
        
        # Combine processed clusters
        if processed_clusters:
            final_group = pd.concat(processed_clusters)
            final_group['timeslot'] = timeslot
            clustered.append(final_group)
            logger.debug(f"Created {len(processed_clusters)} sub-clusters for timeslot {timeslot}")
    
    return pd.concat(clustered) if clustered else pd.DataFrame()

def assign_vehicles(clustered_df, vehicles_df, store_coords):
    """Vehicle assignment with capacity and range validation"""
    trips = []
    vehicle_pool = vehicles_df.copy()
    
    # Sort vehicles by type priority (3W -> 4W-EV -> 4W)
    vehicle_pool['Vehicle Type'] = pd.Categorical(
        vehicle_pool['Vehicle Type'].astype(str),
        categories=['3W', '4W-EV', '4W'], 
        ordered=True
    )
    vehicle_pool.sort_values(['Vehicle Type'], ascending=[True], inplace=True)
    logger.debug(f"Vehicle pool: {vehicle_pool.to_dict('records')}")
    
    def try_assign_vehicle(cluster_df, vehicle_type_priority=0):
        """Try to assign vehicles to cluster, splitting if necessary"""
        if cluster_df.empty:
            return True
            
        available_vehicles = vehicle_pool[vehicle_pool['Vehicle Type'].cat.codes == vehicle_type_priority]
        if available_vehicles.empty and vehicle_type_priority < 2:
            # Try next vehicle type if current type is exhausted
            return try_assign_vehicle(cluster_df, vehicle_type_priority + 1)
        elif available_vehicles.empty:
            return False
            
        vehicle = available_vehicles.iloc[0]
        num_shipments = len(cluster_df)
        
        # Calculate required radius
        max_dist_between_points = 0
        for i, row1 in cluster_df.iterrows():
            for j, row2 in cluster_df.iterrows():
                if i != j:
                    dist = haversine(
                        (row1['Latitude'], row1['Longitude']),
                        (row2['Latitude'], row2['Longitude'])
                    )
                    max_dist_between_points = max(max_dist_between_points, dist)
        
        required_radius = (cluster_df['distance_to_store'].max() + max_dist_between_points) * 1.2
        
        if num_shipments <= vehicle['Shipments_Capacity'] and required_radius <= vehicle['Max Trip Radius (in KM)']:
            # Vehicle can handle entire cluster
            trips.append({
                'shipments': cluster_df['Shipment ID'].tolist(),
                'vehicle_type': vehicle['Vehicle Type'],
                'capacity': vehicle['Shipments_Capacity'],
                'max_distance': vehicle['Max Trip Radius (in KM)'],
                'timeslot': cluster_df['timeslot'].iloc[0],
                'required_radius': required_radius
            })
            idx = vehicle_pool[vehicle_pool['Vehicle Type'] == vehicle['Vehicle Type']].index[0]
            vehicle_pool.at[idx, 'Number'] -= 1
            if vehicle_pool.at[idx, 'Number'] <= 0:
                vehicle_pool.drop(idx, inplace=True)
            logger.debug(f"Assigned {vehicle['Vehicle Type']} to {num_shipments} shipments")
            return True
        elif vehicle_type_priority < 2:
            # Try splitting for smaller vehicles or try larger vehicle
            if num_shipments > vehicle['Shipments_Capacity']:
                # Split cluster and try with same vehicle type
                mid = len(cluster_df) // 2
                return (try_assign_vehicle(cluster_df.iloc[:mid], vehicle_type_priority) and
                       try_assign_vehicle(cluster_df.iloc[mid:], vehicle_type_priority))
            else:
                # Try next vehicle type
                return try_assign_vehicle(cluster_df, vehicle_type_priority + 1)
        else:
            # Split for largest vehicle type
            if num_shipments > vehicle['Shipments_Capacity']:
                mid = len(cluster_df) // 2
                return (try_assign_vehicle(cluster_df.iloc[:mid], vehicle_type_priority) and
                       try_assign_vehicle(cluster_df.iloc[mid:], vehicle_type_priority))
            return False
    
    # Process each cluster
    for timeslot, ts_group in clustered_df.groupby('timeslot'):
        for cluster_id in ts_group['cluster'].unique():
            cluster = ts_group[ts_group['cluster'] == cluster_id].copy()
            cluster['distance_to_store'] = cluster.apply(
                lambda row: haversine(store_coords, (row['Latitude'], row['Longitude'])),
                axis=1
            )
            
            if not try_assign_vehicle(cluster):
                logger.warning(f"Failed to assign vehicles for cluster {cluster_id} in timeslot {timeslot}")
    
    return trips

def optimize_route(store_coords, shipment_coords, vehicle_max_distance):
    """Route optimization with OR-Tools"""
    if not shipment_coords: 
        logger.warning("No shipment coordinates provided")
        return [], 0, []

    try:
        locations = [store_coords] + shipment_coords
        num_locations = len(locations)
        logger.debug(f"Optimizing route for {num_locations} locations")
        
        # Create distance matrix
        distance_matrix = np.zeros((num_locations, num_locations))
        for i in range(num_locations):
            for j in range(num_locations):
                distance_matrix[i][j] = haversine(locations[i], locations[j]) * 1000
        
        manager = pywrapcp.RoutingIndexManager(num_locations, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            int(vehicle_max_distance * 1000 * 1.1),  # vehicle maximum travel distance with 10% buffer
            True,  # start cumul to zero
            dimension_name)

        # Set first solution strategy
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.time_limit.FromSeconds(10)

        solution = routing.SolveWithParameters(search_parameters)
        
        if not solution:
            logger.warning("No solution found")
            return [], 0, []

        # Extract the route
        route_indices = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:  # Skip depot
                route_indices.append(node_index - 1)  # Adjust index to match shipment coords
            index = solution.Value(routing.NextVar(index))
        
        # Calculate leg-by-leg distances for a continuous path
        leg_distances = []
        total_distance = 0
        
        # First leg: Warehouse to first stop
        first_stop_idx = route_indices[0] + 1  # +1 because route_indices are 0-based
        first_leg_dist = distance_matrix[0][first_stop_idx] / 1000
        total_distance += first_leg_dist * 1000
        leg_distances.append({
            'from': "Warehouse",
            'to': f"Stop 1",
            'distance': first_leg_dist
        })
        
        # Middle legs: Between stops
        for i in range(len(route_indices)-1):
            curr_idx = route_indices[i] + 1
            next_idx = route_indices[i+1] + 1
            leg_dist = distance_matrix[curr_idx][next_idx] / 1000
            total_distance += leg_dist * 1000
            leg_distances.append({
                'from': f"Stop {i+1}",
                'to': f"Stop {i+2}",
                'distance': leg_dist
            })
        
        # Final leg: Last stop back to warehouse
        last_stop_idx = route_indices[-1] + 1
        final_leg_dist = distance_matrix[last_stop_idx][0] / 1000
        total_distance += final_leg_dist * 1000
        leg_distances.append({
            'from': f"Stop {len(route_indices)}",
                'to': "Warehouse",
                'distance': final_leg_dist
            })
            
            # Log the continuous route
        logger.debug(f"Route breakdown (continuous path):")
        for leg in leg_distances:
            logger.debug(f"  {leg['from']} → {leg['to']}: {leg['distance']:.2f} km")
        logger.debug(f"Total distance: {total_distance/1000:.2f} km")
            
        return route_indices, total_distance / 1000, leg_distances
    
    except Exception as e:
        logger.error(f"Error in route optimization: {str(e)}")
        return [], 0, []

def calculate_mst(coords, store_coords):
    """Calculate Minimum Spanning Tree distance including depot"""
    if not coords:
        return 0
        
    # Include store location as the first point
    all_points = [store_coords] + coords
    
    # Create complete distance matrix
    n_points = len(all_points)
    dist_matrix = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = haversine(all_points[i], all_points[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    # Calculate MST
    mst = minimum_spanning_tree(dist_matrix)
    mst_distance = mst.sum()
    
    # Add return trip to warehouse from farthest point
    if len(coords) > 0:
        distances_to_warehouse = [haversine(store_coords, coord) for coord in coords]
        return_distance = max(distances_to_warehouse)
        total_distance = mst_distance + return_distance
        logger.debug(f"MST distance: {mst_distance:.2f} km + return trip: {return_distance:.2f} km = {total_distance:.2f} km")
        return total_distance
    return mst_distance

def generate_output_table(trips, shipments_df, store_coords):
    """Generate tabular output with metrics"""
    output = []
    for trip in trips:
        cluster = shipments_df[shipments_df['Shipment ID'].isin(trip['shipments'])]
        coords = cluster[['Latitude', 'Longitude']].values.tolist()
        
        mst_dist = calculate_mst(coords, store_coords)
        trip_time = (mst_dist * TRAVEL_TIME_PER_KM) + (len(cluster) * DELIVERY_TIME_PER_SHIPMENT)
        capacity_uti = (len(cluster) / trip['capacity']) * 100
        
        # Format leg-by-leg distances showing proper sequence
        leg_distances = trip.get('leg_distances', [])
        distance_breakdown = '\n'.join([
            f"{i+1}. {leg['from']} → {leg['to']}: {leg['distance']:.2f} km"
            for i, leg in enumerate(leg_distances)
        ]) if leg_distances else "Not available"
        
        output.append({
            'Timeslot': str(trip['timeslot']),
            'VehicleType': str(trip['vehicle_type']),
            'Shipments': ', '.join(str(s) for s in trip['shipments']),
            'Total Distance (km)': round(trip.get('total_distance', 0), 2),
            'Distance Breakdown': distance_breakdown,
            'MST Distance (km)': round(mst_dist, 2),
            'Estimated Time (min)': round(trip_time, 2),
            'Capacity Utilization (%)': round(capacity_uti, 1)
        })
        
        logger.debug(f"""Trip metrics:
        Vehicle: {trip['vehicle_type']}
        Timeslot: {trip['timeslot']}
        Total Distance: {trip.get('total_distance', 0):.2f} km
        Distance Breakdown:
        {distance_breakdown}
        MST Distance: {mst_dist:.2f} km
        Time: {trip_time:.0f} min
        Utilization: {capacity_uti:.1f}%
        """)
    
    return output

def generate_map(store_coords, trips, shipments_df, show_all=True):
    """Generate interactive Folium map with routes"""
    m = folium.Map(location=store_coords, zoom_start=12)
    
    # Add warehouse marker
    folium.Marker(
        store_coords,
        tooltip='Warehouse',
        icon=folium.Icon(color='green', icon='warehouse', prefix='fa')
    ).add_to(m)

    if show_all:
        for _, shipment in shipments_df.iterrows():
            folium.Marker(
                location=[shipment['Latitude'], shipment['Longitude']],
                popup=f"Shipment {shipment['Shipment ID']}",
                icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')
            ).add_to(m)

    # Color palette for routes
    colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple',
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    for trip_idx, trip in enumerate(trips):
        shipment_ids = trip.get('route_order_ids', trip['shipments'])
        cluster = shipments_df[shipments_df['Shipment ID'].isin(shipment_ids)]
        
        # Create route coordinates in correct sequence
        route_coords = [store_coords]  # Start with warehouse
        for i, sid in enumerate(shipment_ids, 1):
            shipment = cluster[cluster['Shipment ID'] == sid].iloc[0]
            route_coords.append([shipment['Latitude'], shipment['Longitude']])
        route_coords.append(store_coords)  # Return to warehouse
        
        # Draw route polyline
        color = colors[trip_idx % len(colors)]
        folium.PolyLine(
            route_coords,
            color=color,
            weight=2.5,
            opacity=0.8,
            tooltip=f"{trip['vehicle_type']} - {trip['timeslot']}"
        ).add_to(m)
        
        # Add sequence number markers in order
        for seq, sid in enumerate(shipment_ids, 1):
            shipment = cluster[cluster['Shipment ID'] == sid].iloc[0]
            folium.Marker(
                [shipment['Latitude'], shipment['Longitude']],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12pt; color: {color};">{seq}</div>'
                ),
                tooltip=f"Stop {seq}: Shipment {sid}"
            ).add_to(m)
            
            # Add shopping cart marker for delivery spots
            if not show_all:
                folium.Marker(
                    location=[shipment['Latitude'], shipment['Longitude']],
                    popup=f"Shipment {shipment['Shipment ID']}",
                    icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')
                ).add_to(m)
    
    return m._repr_html_()

def generate_excel_output(trips, shipments_df, store_coords):
    """Generate Excel output with detailed metrics"""
    output_rows = []
    
    for trip_idx, trip in enumerate(trips, 1):
        cluster = shipments_df[shipments_df['Shipment ID'].isin(trip['shipments'])]
        coords = cluster[['Latitude', 'Longitude']].values.tolist()
        
        mst_dist = calculate_mst(coords, store_coords)
        trip_time = (mst_dist * TRAVEL_TIME_PER_KM) + (len(cluster) * DELIVERY_TIME_PER_SHIPMENT)
        capacity_uti = (len(cluster) / trip['capacity']) * 100
        time_uti = trip_time / (8 * 60) * 100  # Assuming 8-hour workday
        coverage_uti = (mst_dist / trip['max_distance']) * 100
        
        # Create individual rows for each shipment in the trip
        for _, shipment in cluster.iterrows():
            output_rows.append({
                'TRIP ID': f"Trip_{trip_idx}",
                'Shipment ID': shipment['Shipment ID'],
                'Latitude': shipment['Latitude'],
                'Longitude': shipment['Longitude'],
                'TIME SLOT': trip['timeslot'],
                'Shipments': len(trip['shipments']),
                'MST_DIST': round(mst_dist, 2),
                'TRIP_TIME': round(trip_time, 2),
                'Vehicle_Type': trip['vehicle_type'],
                'CAPACITY_UTI': round(capacity_uti, 2),
                'TIME_UTI': round(time_uti, 2),
                'COV_UTI': round(coverage_uti, 2)
            })
    
    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(output_rows)
    df.to_excel('route_optimization_results.xlsx', index=False)
    return df

@app.route('/')
def optimize_routes():
    try:
        shipments_df, vehicles_df, store_coords = load_data()
        logger.info("Data loaded successfully")
        
        clustered_df = cluster_shipments(shipments_df, store_coords, vehicles_df)
        if clustered_df.empty:
            logger.warning("No shipments clustered")
            return render_template('results.html', 
                                 output_table=[],
                                 map_html='<div>No shipments to process</div>')
        
        trips = assign_vehicles(clustered_df, vehicles_df, store_coords)
        logger.info(f"Assigned {len(trips)} trips to vehicles")
        
        optimized_trips = []
        for idx, trip in enumerate(trips):
            logger.debug(f"Optimizing trip {idx+1}/{len(trips)}")
            cluster = shipments_df[shipments_df['Shipment ID'].isin(trip['shipments'])]
            coords = cluster[['Latitude', 'Longitude']].values.tolist()
            
            route, distance, leg_distances = optimize_route(
                store_coords,
                coords,
                trip['max_distance']
            )
            
            if route and distance > 0:
                trip['route_order_ids'] = [trip['shipments'][i] for i in route]
                trip['total_distance'] = distance
                trip['leg_distances'] = leg_distances  # Add leg distances to trip
                optimized_trips.append(trip)
                logger.debug(f"Trip {idx+1} optimized successfully")
            else:
                logger.warning(f"Failed to optimize trip {idx+1}")
        
        if not optimized_trips:
            logger.warning("No trips were successfully optimized")
            return render_template('results.html', 
                                 output_table=[],
                                 map_html='<div>No valid routes found. Please check shipment locations and vehicle constraints.</div>')
        
        # Generate output
        output_table = generate_output_table(optimized_trips, shipments_df, store_coords)
        if not output_table:  # Add check for empty output
            return render_template('results.html', 
                                 output_table=[],
                                 map_html='<div>No valid routes found</div>')
            
        map_html = generate_map(store_coords, optimized_trips, shipments_df)
        
        # Generate Excel output
        excel_output = generate_excel_output(optimized_trips, shipments_df, store_coords)
        logger.info("Excel output generated successfully")
        
        return render_template('results.html', 
                             output_table=output_table,
                             map_html=map_html)
    
    except Exception as e:
        logger.error(f"Error in optimize_routes: {str(e)}")
        import traceback
        return f"Error processing request: {traceback.format_exc()}", 500

@app.route('/show_trip/<int:trip_index>')
def show_trip(trip_index):
    try:
        shipments_df, vehicles_df, store_coords = load_data()
        clustered_df = cluster_shipments(shipments_df, store_coords, vehicles_df)
        trips = assign_vehicles(clustered_df, vehicles_df, store_coords)
        
        optimized_trips = []
        for idx, trip in enumerate(trips):
            cluster = shipments_df[shipments_df['Shipment ID'].isin(trip['shipments'])]
            coords = cluster[['Latitude', 'Longitude']].values.tolist()
            
            route, distance, leg_distances = optimize_route(
                store_coords,
                coords,
                trip['max_distance']
            )
            
            if route and distance > 0:
                trip['route_order_ids'] = [trip['shipments'][i] for i in route]
                trip['total_distance'] = distance
                trip['leg_distances'] = leg_distances
                optimized_trips.append(trip)
        
        if trip_index < len(optimized_trips):
            trip = optimized_trips[trip_index]
            map_html = generate_map(store_coords, [trip], shipments_df, show_all=False)
            return map_html
        else:
            return '<div>Invalid trip index</div>', 400
    
    except Exception as e:
        logger.error(f"Error in show_trip: {str(e)}")
        import traceback
        return f"Error processing request: {traceback.format_exc()}", 500

if __name__ == '__main__':
    app.run(debug=True)


