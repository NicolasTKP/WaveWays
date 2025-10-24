import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import {
  Ship,
  TrendingDown,
  Clock,
  Navigation,
  Anchor,
  AlertCircle,
  CheckCircle2,
  ArrowLeft,
  Loader2,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { RouteMap } from "@/components/RouteMap";
import { Port } from "@/types/route"; // Keep Port type
import { useToast } from "@/hooks/use-toast";
import {
  initializeMultiLegSimulation,
  PointModel,
  VesselConfig,
} from "@/services/routeService";

interface RouteVisualizationState {
  sessionID: string;
  vesselConfig: VesselConfig;
  sequencedDestinations: PointModel[];
  startPoint: Port;
}

interface OptimizedRouteData {
  fullAstarPath: [number, number][];
  landmarkPoints: [number, number][];
  initialVesselState: {
    lat: number;
    lon: number;
    speed: number;
    heading: number;
  };
  // Add other metrics if needed from get_next_action or initial simulation
  totalDistance?: number;
  totalFuel?: number;
  totalTime?: number;
  warnings?: string[];
}

const RouteVisualization = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const location = useLocation();
  const { sessionID, vesselConfig, sequencedDestinations, startPoint } =
    (location.state as RouteVisualizationState) || {};

  const [optimizedRouteData, setOptimizedRouteData] =
    useState<OptimizedRouteData | null>(null);
  const [isCalculating, setIsCalculating] = useState(true);

  // The current location will be the startPoint initially, then updated by DRL simulation
  const [currentLocationState, setCurrentLocationState] = useState<Port | null>(
    startPoint || null
  );
  const [totalDistance, setTotalDistance] = useState<number>(0);
  const [totalFuel, setTotalFuel] = useState<number>(0);
  const [totalTime, setTotalTime] = useState<number>(0);
  const [warnings, setWarnings] = useState<string[]>([]);

  // Helper to convert degrees to radians
  const toRad = (deg: number) => deg * (Math.PI / 180);

  // Helper to calculate Haversine distance between two points (lat, lon) in nautical miles
  const calculateDistance = (
    lat1: number,
    lon1: number,
    lat2: number,
    lon2: number
  ): number => {
    const R = 3440.065; // Earth radius in nautical miles
    const dLat = toRad(lat2 - lat1);
    const dLon = toRad(lon2 - lon1);
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(toRad(lat1)) *
        Math.cos(toRad(lat2)) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  // Convert sequencedDestinations (PointModel[]) to Port[] for RouteMap
  const portsForMap: Port[] = [
    startPoint,
    ...sequencedDestinations.map((p, index) => ({
      id: `seq-${index}`,
      name: `Destination ${index + 1}`,
      lat: p.lat,
      lng: p.lon,
      country: "Unknown",
      coordinates: `${p.lat},${p.lon}`,
    })),
  ];

  useEffect(() => {
    if (!sessionID || !vesselConfig || !sequencedDestinations || !startPoint) {
      toast({
        title: "No Route Data",
        description: "Please configure your route first.",
        variant: "destructive",
      });
      navigate("/plan-route");
      return;
    }

    const initializeSimulation = async () => {
      setIsCalculating(true);
      try {
        const requestBody = {
          session_id: sessionID,
          start_point: {
            lat: startPoint.lat,
            lon: startPoint.lng,
            name: startPoint.name, // Include the name property
          },
          sequenced_destinations: sequencedDestinations, // Already PointModel[] with name
          initial_speed: vesselConfig.speed_knots,
          initial_heading: 0, // Assuming initial heading is 0 for now
          vessel_config: {
            speed_knots: vesselConfig.speed_knots,
            fuel_consumption_t_per_day: vesselConfig.fuel_consumption_t_per_day,
            fuel_tank_capacity_t: vesselConfig.fuel_tank_capacity_t,
            safety_fuel_margin_t: vesselConfig.safety_fuel_margin_t,
          },
        };

        const response = await initializeMultiLegSimulation(requestBody);

        setOptimizedRouteData({
          fullAstarPath: response.full_astar_path,
          landmarkPoints: response.landmark_points,
          initialVesselState: response.initial_vessel_state,
        });
        setCurrentLocationState({
          id: "current",
          name: "Current Position",
          lat: response.initial_vessel_state.lat,
          lng: response.initial_vessel_state.lon,
          country: "Ocean",
          coordinates: `${response.initial_vessel_state.lat},${response.initial_vessel_state.lon}`,
        });

        toast({
          title: "Simulation Initialized",
          description: "Multi-leg route simulation is ready.",
        });
      } catch (error: any) {
        toast({
          title: "Error Initializing Simulation",
          description:
            error.message || "Failed to initialize multi-leg simulation.",
          variant: "destructive",
        });
        navigate("/plan-route");
      } finally {
        setIsCalculating(false);
      }
    };

    initializeSimulation();
  }, [
    sessionID,
    vesselConfig,
    sequencedDestinations,
    startPoint,
    navigate,
    toast,
  ]);

  // Calculate total distance, fuel, time once optimizedRouteData is available
  useEffect(() => {
    if (optimizedRouteData?.fullAstarPath && vesselConfig) {
      let calculatedDistance = 0;
      for (let i = 0; i < optimizedRouteData.fullAstarPath.length - 1; i++) {
        const [lat1, lon1] = optimizedRouteData.fullAstarPath[i];
        const [lat2, lon2] = optimizedRouteData.fullAstarPath[i + 1];
        calculatedDistance += calculateDistance(lat1, lon1, lat2, lon2);
      }
      setTotalDistance(calculatedDistance);

      const calculatedTravelTime =
        calculatedDistance / vesselConfig.speed_knots;
      setTotalTime(calculatedTravelTime);

      const calculatedFuelConsumption =
        (calculatedTravelTime / 24) * vesselConfig.fuel_consumption_t_per_day;
      setTotalFuel(calculatedFuelConsumption);

      if (
        calculatedFuelConsumption >
        vesselConfig.fuel_tank_capacity_t - vesselConfig.safety_fuel_margin_t
      ) {
        setWarnings([
          "⚠️ Fuel capacity may be insufficient for this route. Consider adding refueling stops.",
        ]);
      } else {
        setWarnings([]);
      }
    }
  }, [optimizedRouteData, vesselConfig]);

  const handleMapClick = (lat: number, lng: number) => {
    // This function will be used for D* Lite rerouting later
    // For now, just log or show a toast
    toast({
      title: "Map Clicked",
      description: `Lat: ${lat.toFixed(4)}, Lng: ${lng.toFixed(4)}`,
    });
  };

  if (isCalculating) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <Loader2 className="h-16 w-16 text-primary animate-spin mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2">
            Initializing Route Simulation
          </h2>
          <p className="text-muted-foreground">
            Please wait while we prepare your optimized maritime route...
          </p>
        </div>
      </div>
    );
  }

  if (!optimizedRouteData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <AlertCircle className="h-16 w-16 text-destructive mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2">No Route Data Available</h2>
          <p className="text-muted-foreground">
            Something went wrong. Please go back to the planning page.
          </p>
          <Button onClick={() => navigate("/plan-route")} className="mt-4">
            Go to Plan Route
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />

      <main className="flex-1">
        {/* Header */}
        <section className="py-12 bg-gradient-to-b from-primary/5 to-background border-b">
          <div className="container mx-auto px-4">
            <div className="flex items-center justify-between mb-6">
              <Button
                variant="ghost"
                onClick={() => navigate("/plan-route")}
                className="gap-2"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to Planning
              </Button>
              <Badge variant="outline" className="gap-2 px-4 py-2">
                <CheckCircle2 className="h-4 w-4 text-accent" />
                Route Optimized
              </Badge>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              Your Optimized Maritime Route
            </h1>
            <p className="text-lg text-muted-foreground">
              Route from{" "}
              <span className="font-semibold text-foreground">
                {startPoint?.name}
              </span>{" "}
              to{" "}
              <span className="font-semibold text-foreground">
                {portsForMap[portsForMap.length - 1]?.name}
              </span>
            </p>
          </div>
        </section>

        {/* Warnings */}
        {warnings.length > 0 && (
          <section className="py-6 bg-destructive/5 border-b border-destructive/10">
            <div className="container mx-auto px-4">
              {warnings.map((warning, i) => (
                <div
                  key={i}
                  className="flex items-start gap-3 p-4 rounded-lg bg-background border border-destructive/20 max-w-4xl"
                >
                  <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-destructive font-medium">
                    {warning}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Main Content - All Visible Together */}
        <section className="py-12">
          <div className="container mx-auto px-4">
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
              {/* Map - Takes 2/3 width on large screens */}
              <Card className="xl:col-span-2 border-2 shadow-lg">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Navigation className="h-5 w-5 text-primary" />
                    <CardTitle>Route Map</CardTitle>
                  </div>
                  <CardDescription>
                    Interactive map showing your optimized route
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <RouteMap
                    ports={portsForMap}
                    route={[
                      {
                        start: startPoint,
                        end: portsForMap[portsForMap.length - 1],
                        path: optimizedRouteData.fullAstarPath,
                        distance: totalDistance,
                        fuelConsumption: totalFuel,
                      },
                    ]}
                    currentLocation={currentLocationState}
                    onMapClick={handleMapClick} // Enable map clicks for D* Lite
                  />
                </CardContent>
              </Card>

              {/* Metrics & Summary - Takes 1/3 width on large screens */}
              <div className="space-y-6">
                {/* Key Metrics */}
                <Card className="border-2 shadow-lg">
                  <CardHeader>
                    <CardTitle className="text-lg">Key Metrics</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center gap-3 p-3 rounded-lg bg-primary/5 border border-primary/10">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <Navigation className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">
                          Total Distance
                        </p>
                        <p className="text-xl font-bold">
                          {totalDistance
                            ? `${totalDistance.toFixed(1)} nm`
                            : "-- nm"}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 p-3 rounded-lg bg-accent/5 border border-accent/10">
                      <div className="p-2 rounded-lg bg-accent/10">
                        <Clock className="h-5 w-5 text-accent" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">
                          Travel Time
                        </p>
                        <p className="text-xl font-bold">
                          {totalTime ? `${totalTime.toFixed(1)} hrs` : "-- hrs"}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 p-3 rounded-lg bg-muted">
                      <div className="p-2 rounded-lg bg-background">
                        <TrendingDown className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">
                          Fuel Usage
                        </p>
                        <p className="text-xl font-bold">
                          {totalFuel
                            ? `${totalFuel.toFixed(1)} tons`
                            : "-- tons"}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 p-3 rounded-lg bg-muted">
                      <div className="p-2 rounded-lg bg-background">
                        <Anchor className="h-5 w-5 text-muted-foreground" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">
                          Total Ports
                        </p>
                        <p className="text-xl font-bold">
                          {portsForMap.length}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Route Summary */}
                {optimizedRouteData && (
                  <Card className="border-2 shadow-lg">
                    <CardHeader>
                      <CardTitle className="text-lg">Route Summary</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-muted-foreground">
                          Origin
                        </p>
                        <div className="p-3 rounded-lg bg-accent/5 border border-accent/10">
                          <p className="font-semibold">{startPoint?.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {startPoint?.country}
                          </p>
                        </div>
                      </div>

                      {portsForMap.length > 2 && (
                        <div className="space-y-2">
                          <p className="text-sm font-medium text-muted-foreground">
                            Waypoints
                          </p>
                          <div className="space-y-2">
                            {portsForMap.slice(1, -1).map((port, idx) => (
                              <div
                                key={idx}
                                className="p-2 rounded-lg bg-muted text-sm"
                              >
                                <p className="font-medium">{port.name}</p>
                                <p className="text-xs text-muted-foreground">
                                  {port.country}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="space-y-2">
                        <p className="text-sm font-medium text-muted-foreground">
                          Destination
                        </p>
                        <div className="p-3 rounded-lg bg-primary/5 border border-primary/10">
                          <p className="font-semibold">
                            {portsForMap[portsForMap.length - 1]?.name}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {portsForMap[portsForMap.length - 1]?.country}
                          </p>
                        </div>
                      </div>

                      <div className="pt-4 border-t space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">
                            Route Segments
                          </span>
                          <span className="font-semibold">
                            {optimizedRouteData.fullAstarPath.length > 0
                              ? 1
                              : 0}{" "}
                            {/* A single segment for the full A* path */}
                          </span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">
                            Avg Speed
                          </span>
                          <span className="font-semibold">
                            {vesselConfig?.speed_knots} knots
                          </span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">
                            Fuel Efficiency
                          </span>
                          <span className="font-semibold">
                            {totalFuel && totalDistance
                              ? ((totalFuel / totalDistance) * 100).toFixed(2)
                              : "--"}{" "}
                            tons/100nm
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default RouteVisualization;
