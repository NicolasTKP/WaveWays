export interface VesselConfig {
  speed: number;
  fuelConsumption: number;
  tankCapacity: number;
  safetyMargin: number;
  length: number;
  width: number;
  height: number;
  underwaterPercent: number;
}

export interface Port {
  id: string;
  name: string;
  lat: number;
  lng: number;
  country?: string;
  coordinates: string;
}

export interface RouteSegment {
  start: Port;
  end: Port;
  path: [number, number][];
  distance: number;
  fuelConsumption: number;
  travelTime: number;
  weatherConditions?: {
    waveHeight: number;
    windSpeed: number;
    penalty: number;
  };
}

export interface OptimizedRoute {
  segments: RouteSegment[];
  totalDistance: number;
  totalFuel: number;
  totalTime: number;
  ports: Port[];
  warnings: string[];
}

export interface RouteRequest {
  vessel: VesselConfig;
  ports: Port[];
  optimizationMode?: 'fuel' | 'time' | 'balanced';
}

export interface RouteResponse {
  success: boolean;
  route?: OptimizedRoute;
  error?: string;
}
