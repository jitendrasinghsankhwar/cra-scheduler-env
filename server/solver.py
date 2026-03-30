"""OR-Tools VRPTW solver for computing optimal CRA routes."""

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

try:
    from ..distances import get_distance, get_travel_days
except ImportError:
    from distances import get_distance, get_travel_days


def solve(task: dict) -> dict:
    """Solve the VRPTW problem for a task. Returns optimal routes and total cost.

    Only feasible for tasks with <= 20 sites (OR-Tools gets slow beyond that).
    Returns None if no feasible solution found.
    """
    cras = task["cras"]
    sites = task["sites"]
    num_cras = len(cras)
    num_sites = len(sites)
    num_nodes = num_cras + num_sites

    # Map node index to city name
    node_city = {}
    for i, cra in enumerate(cras):
        node_city[i] = cra["home_city"]
    for i, site in enumerate(sites):
        node_city[num_cras + i] = site["name"]

    depot_indices = list(range(num_cras))

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return get_distance(node_city[from_node], node_city[to_node])

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return get_travel_days(node_city[from_node], node_city[to_node])

    manager = pywrapcp.RoutingIndexManager(
        num_nodes, num_cras, depot_indices, depot_indices,
    )
    routing = pywrapcp.RoutingModel(manager)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    max_days = task.get("max_days", 60)
    routing.AddDimension(time_callback_index, max_days, max_days, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    for i, site in enumerate(sites):
        node_index = manager.NodeToIndex(num_cras + i)
        time_dimension.CumulVar(node_index).SetRange(
            site["window_start"], site["window_end"],
        )

    for i in range(num_cras):
        start_index = routing.Start(i)
        time_dimension.CumulVar(start_index).SetRange(1, 1)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 10

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return None

    routes = {}
    total_cost = 0
    for cra_idx in range(num_cras):
        route = []
        index = routing.Start(cra_idx)
        route_cost = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            next_index = solution.Value(routing.NextVar(index))
            route_cost += distance_callback(index, next_index)
            if node >= num_cras:
                route.append(sites[node - num_cras]["name"])
            index = next_index
        routes[cra_idx] = route
        total_cost += route_cost

    return {"routes": routes, "total_cost": total_cost}
