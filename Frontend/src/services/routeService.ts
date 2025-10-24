import { Port } from "@/types/route";

const API_BASE_URL = "http://127.0.0.1:8000/api";

export interface VesselConfig {
  speed_knots: number;
  fuel_consumption_t_per_day: number;
  fuel_tank_capacity_t: number;
  safety_fuel_margin_t: number;
  length: number;
  width: number;
  height: number;
  underwater_percent: number;
}

export interface PointModel {
  lat: number;
  lon: number;
  name: string; // Add the name property
}

export interface UnreachableDestination {
  lat: number;
  lon: number;
  reason: string;
}

export interface SuggestRouteSequenceRequest {
  start_point: PointModel;
  destinations: PointModel[];
  is_cycle_route: boolean;
  vessel_config: {
    speed_knots: number;
    fuel_consumption_t_per_day: number;
    fuel_tank_capacity_t: number;
    safety_fuel_margin_t: number;
  };
}

export interface SuggestedRouteSequenceResponse {
  session_id: string;
  suggested_sequence: PointModel[];
  unreachable_destinations: UnreachableDestination[];
}

export interface InitializeMultiLegSimulationRequest {
  session_id: string;
  start_point: PointModel;
  sequenced_destinations: PointModel[];
  initial_speed: number;
  initial_heading: number;
  vessel_config: {
    speed_knots: number;
    fuel_consumption_t_per_day: number;
    fuel_tank_capacity_t: number;
    safety_fuel_margin_t: number;
  };
}

export interface InitializeMultiLegSimulationResponse {
  session_id: string;
  message: string;
  initial_vessel_state: {
    lat: number;
    lon: number;
    speed: number;
    heading: number;
  };
  full_astar_path: [number, number][];
  landmark_points: [number, number][];
  unreachable_destinations: { lat: number; lon: number; reason: string; }[];
  warnings: string[];
}

export const suggestRouteSequence = async (
  request: SuggestRouteSequenceRequest
): Promise<SuggestedRouteSequenceResponse> => {
  const response = await fetch(`${API_BASE_URL}/suggest_route_sequence`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to suggest route sequence");
  }

  return response.json();
};

export const initializeMultiLegSimulation = async (
  request: InitializeMultiLegSimulationRequest
): Promise<InitializeMultiLegSimulationResponse> => {
  const response = await fetch(`${API_BASE_URL}/initialize_multi_leg_simulation`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Failed to initialize multi-leg simulation");
  }

  return response.json();
};
