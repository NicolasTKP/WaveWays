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
  getNextAction, // Import getNextAction
  GetNextActionResponse, // Import GetNextActionResponse
  getDStarLiteRoute, // Import getDStarLiteRoute
  ObstaclePolygonModel, // Import ObstaclePolygonModel
  predictETA, // Import predictETA
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
  unreachableDestinations?: { lat: number; lon: number; reason: string }[];
  imagePath?: string; // New field for the path to the saved visualization image
}

const RouteVisualization = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const location = useLocation();
  const { sessionID, vesselConfig, sequencedDestinations, startPoint } =
    (location.state as RouteVisualizationState) || {};

  const [optimizedRouteData, setOptimizedRouteData] =
    useState<OptimizedRouteData | null>(null);
  const [isLoading, setIsLoading] = useState(true); // Changed from isCalculating

  // The current location will be the startPoint initially, then updated by DRL simulation
  const [currentLocationState, setCurrentLocationState] = useState<Port | null>(
    startPoint || null
  );
  const [totalDistance, setTotalDistance] = useState<number>(0);
  const [totalFuel, setTotalFuel] = useState<number>(0);
  const [totalTime, setTotalTime] = useState<number>(0);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [eta, setEta] = useState<string | null>(null); // New state for ETA

  // New states for DRL simulation
  const [suggestedSpeed, setSuggestedSpeed] = useState<number | null>(null);
  const [suggestedHeading, setSuggestedHeading] = useState<number | null>(null);
  const [currentVesselSpeed, setCurrentVesselSpeed] = useState<number | null>(
    null
  );
  const [currentVesselHeading, setCurrentVesselHeading] = useState<
    number | null
  >(null);
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [simulationDone, setSimulationDone] = useState(false);
  const [dStarLitePath, setDStarLitePath] = useState<[number, number][] | null>(
    null
  );
  const [obstaclePolygon, setObstaclePolygon] = useState<PointModel[]>([]);

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

  // Helper to generate random initial speed and heading
  const generateRandomInitialVesselState = (maxSpeed: number) => {
    const randomSpeed = Math.random() * (maxSpeed * 0.5) + maxSpeed * 0.5; // 50-100% of max speed
    const randomHeading = Math.random() * 360; // 0-359 degrees
    return { speed: randomSpeed, heading: randomHeading };
  };

  // Function to call get_next_action API
  const handleGetNextAction = async (
    lat: number,
    lon: number,
    speed: number,
    heading: number
  ) => {
    if (!sessionID) {
      toast({
        title: "Error",
        description: "Session ID is missing for next action.",
        variant: "destructive",
      });
      return;
    }
    setIsLoading(true);
    try {
      const requestBody = {
        session_id: sessionID,
        current_lat: lat,
        current_lon: lon,
        current_speed: speed,
        current_heading: heading,
      };
      const response: GetNextActionResponse = await getNextAction(requestBody);

      setSuggestedSpeed(response.new_speed);
      setSuggestedHeading(response.new_heading);
      setCurrentVesselSpeed(response.new_speed); // Update for next step
      setCurrentVesselHeading(response.new_heading); // Update for next step
      setSimulationDone(response.done);

      // Update current location state with new lat/lon from API
      setCurrentLocationState({
        id: "current",
        name: "Current Position",
        lat: response.new_lat,
        lng: response.new_lon,
        country: "Ocean",
        coordinates: `${response.new_lat},${response.new_lon}`,
      });

      toast({
        title: "Next Action Suggested",
        description: `Speed: ${response.new_speed.toFixed(
          1
        )} knots, Heading: ${response.new_heading.toFixed(1)}°`,
      });

      if (response.done) {
        toast({
          title: "Simulation Complete",
          description:
            "The vessel has reached its destination or run out of fuel.",
          variant: "default", // Changed from "success" to "default"
        });
        setIsSimulationRunning(false);
      }
    } catch (error: any) {
      toast({
        title: "Error Getting Next Action",
        description:
          error.message || "Failed to get next action from DRL agent.",
        variant: "destructive",
      });
      setIsSimulationRunning(false);
    } finally {
      setIsLoading(false);
    }
  };

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
      setIsLoading(true); // Changed from setIsCalculating
      try {
        const requestBody = {
          session_id: sessionID,
          start_point: {
            lat: startPoint.lat,
            lon: startPoint.lng,
            name: startPoint.name, // Include the name property
          },
          sequenced_destinations: sequencedDestinations, // Already PointModel[] with name
          initial_speed: vesselConfig.speed_knots, // Initial speed for env setup
          initial_heading: 0, // Initial heading for env setup
          vessel_config: {
            speed_knots: vesselConfig.speed_knots,
            fuel_consumption_t_per_day: vesselConfig.fuel_consumption_t_per_day,
            fuel_tank_capacity_t: vesselConfig.fuel_tank_capacity_t,
            safety_fuel_margin_t: vesselConfig.safety_fuel_margin_t,
          },
        };

        const response = await initializeMultiLegSimulation(requestBody);

        // Generate random initial speed and heading for the first DRL step
        const { speed: randomInitialSpeed, heading: randomInitialHeading } =
          generateRandomInitialVesselState(vesselConfig.speed_knots);

        setCurrentVesselSpeed(randomInitialSpeed);
        setCurrentVesselHeading(randomInitialHeading);

        // Filter out unreachable destinations from the sequencedDestinations
        const reachableDestinations = sequencedDestinations.filter(
          (dest) =>
            !response.unreachable_destinations.some(
              (unreachable) =>
                unreachable.lat === dest.lat && unreachable.lon === dest.lon
            )
        );

        setOptimizedRouteData({
          fullAstarPath: response.full_astar_path,
          landmarkPoints: response.landmark_points,
          initialVesselState: response.initial_vessel_state,
          unreachableDestinations: response.unreachable_destinations,
          warnings: response.warnings,
          imagePath: response.image_path, // Store the image path
        });
        setCurrentLocationState({
          id: "current",
          name: "Current Position",
          lat: response.initial_vessel_state.lat,
          lng: response.initial_vessel_state.lon,
          country: "Ocean",
          coordinates: `${response.initial_vessel_state.lat},${response.initial_vessel_state.lon}`,
        });

        // Automatically trigger the first DRL action after initialization
        if (response.full_astar_path.length > 0) {
          handleGetNextAction(
            response.initial_vessel_state.lat,
            response.initial_vessel_state.lon,
            randomInitialSpeed,
            randomInitialHeading
          );
        }

        // Display warnings and unreachable destinations as toasts
        response.warnings.forEach((warning) => {
          toast({
            title: "Route Warning",
            description: warning,
            variant: "default", // Use default for warnings, not destructive
          });
        });

        response.unreachable_destinations.forEach((unreachable) => {
          toast({
            title: "Unreachable Destination",
            description: `Destination at Lat: ${unreachable.lat.toFixed(
              4
            )}, Lng: ${unreachable.lon.toFixed(4)} is unreachable. Reason: ${
              unreachable.reason
            }`,
            variant: "destructive",
          });
        });

        if (response.full_astar_path.length > 0) {
          toast({
            title: "Simulation Initialized",
            description: "Multi-leg route simulation is ready.",
          });
        } else {
          toast({
            title: "Simulation Initialized with Issues",
            description:
              "No complete route could be generated. Check unreachable destinations.",
            variant: "destructive",
          });
        }
      } catch (error: any) {
        toast({
          title: "Error Initializing Simulation",
          description:
            error.message || "Failed to initialize multi-leg simulation.",
          variant: "destructive",
        });
        navigate("/plan-route");
      } finally {
        setIsLoading(false); // Changed from setIsCalculating
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
    // Add handleGetNextAction to dependencies if it's stable, or memoize it
    // For now, it's fine as it's defined outside the useEffect and uses stable state setters
  ]);

  // Convert sequencedDestinations (PointModel[]) to Port[] for RouteMap
  // Filter out unreachable destinations from the original sequencedDestinations
  const filteredSequencedDestinations = sequencedDestinations.filter(
    (dest) =>
      !optimizedRouteData?.unreachableDestinations?.some(
        (unreachable) =>
          unreachable.lat === dest.lat && unreachable.lon === dest.lon
      )
  );

  const portsForMap: Port[] = [
    startPoint,
    ...filteredSequencedDestinations.map((p, index) => ({
      id: `seq-${index}`,
      name: p.name || `Destination ${index + 1}`, // Use name from PointModel if available
      lat: p.lat,
      lng: p.lon,
      country: "Unknown",
      coordinates: `${p.lat},${p.lon}`,
    })),
  ];

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

      const allWarnings = [...(optimizedRouteData.warnings || [])]; // Start with backend warnings

      if (
        calculatedFuelConsumption >
        vesselConfig.fuel_tank_capacity_t - vesselConfig.safety_fuel_margin_t
      ) {
        allWarnings.push(
          "⚠️ Fuel capacity may be insufficient for this route. Consider adding refueling stops."
        );
      }
      setWarnings(allWarnings);
    }
  }, [optimizedRouteData, vesselConfig]);

  // Effect to call predictETA when optimizedRouteData and vesselConfig are available
  useEffect(() => {
    const fetchETA = async () => {
      if (optimizedRouteData?.fullAstarPath.length && vesselConfig) {
        const firstPoint = optimizedRouteData.fullAstarPath[0];
        const lastPoint =
          optimizedRouteData.fullAstarPath[
            optimizedRouteData.fullAstarPath.length - 1
          ];

        const recentPoints = [
          {
            MMSI: Math.floor(Math.random() * 1000000000), // Random MMSI
            BaseDateTime: new Date().toISOString(), // Current date time
            LAT: firstPoint[0],
            LON: firstPoint[1],
            SOG: parseFloat((Math.random() * 20).toFixed(2)), // Random Speed Over Ground (0-20 knots)
            COG: parseFloat((Math.random() * 360).toFixed(2)), // Random Course Over Ground (0-360 degrees)
            Heading: parseFloat((Math.random() * 360).toFixed(2)), // Random Heading (0-360 degrees)
            VesselType: 70, // Random Vessel Type (e.g., 70 for Cargo)
            Length: vesselConfig.length,
            Width: vesselConfig.width,
            Draft: parseFloat((Math.random() * 15).toFixed(2)), // Random Draft (0-15 meters)
          },
        ];

        const requestBody = {
          recent_points: recentPoints,
          destination_lat: lastPoint[0],
          destination_lon: lastPoint[1],
        };
        console.log("Request Body for predictETA:", requestBody); // Log request body

        try {
          const response = await predictETA(requestBody);
          console.log("Full API Response for predictETA:", response); // Log entire response
          setEta(response.predicted_arrival_time_utc);
          console.log("Predicted ETA (from response.eta):", response.predicted_arrival_time_utc);
          toast({
            title: "ETA Predicted",
            description: `Estimated Time of Arrival: ${response.predicted_arrival_time_utc}`,
          });
        } catch (error: any) {
          toast({
            title: "Error Predicting ETA",
            description: error.message || "Failed to predict ETA.",
            variant: "destructive",
          });
        }
      }
    };

    fetchETA();
  }, [optimizedRouteData, vesselConfig, toast]);

  const handleMapClick = (lat: number, lng: number) => {
    // Log or show a toast with clicked coordinates
    toast({
      title: "Map Clicked",
      description: `Lat: ${lat.toFixed(4)}, Lng: ${lng.toFixed(4)}`,
    });
    // Trigger handleNextStep on map click
    handleNextStep();
  };

  const handleNextStep = () => {
    if (
      currentLocationState &&
      currentVesselSpeed !== null &&
      currentVesselHeading !== null
    ) {
      handleGetNextAction(
        currentLocationState.lat,
        currentLocationState.lng,
        currentVesselSpeed,
        currentVesselHeading
      );
    } else {
      toast({
        title: "Simulation Not Ready",
        description: "Please wait for initialization or check for errors.",
        variant: "destructive",
      });
    }
  };

  const handleGenerateObstacleAndReroute = async () => {
    if (!sessionID || !optimizedRouteData?.fullAstarPath.length) {
      toast({
        title: "Error",
        description: "Session ID or A* path is missing for rerouting.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      const path = optimizedRouteData.fullAstarPath;
      const midIndex = Math.floor(path.length / 2);
      const midPoint = path[midIndex]; // [lat, lon]

      // Generate a simple square obstacle around the midpoint
      const obstacleSize = 0.1; // degrees
      const obstaclePoints: PointModel[] = [
        {
          lat: midPoint[0] - obstacleSize,
          lon: midPoint[1] - obstacleSize,
          name: "Obstacle_BL",
        },
        {
          lat: midPoint[0] + obstacleSize,
          lon: midPoint[1] - obstacleSize,
          name: "Obstacle_TL",
        },
        {
          lat: midPoint[0] + obstacleSize,
          lon: midPoint[1] + obstacleSize,
          name: "Obstacle_TR",
        },
        {
          lat: midPoint[0] - obstacleSize,
          lon: midPoint[1] + obstacleSize,
          name: "Obstacle_BR",
        },
        {
          lat: midPoint[0] - obstacleSize,
          lon: midPoint[1] - obstacleSize,
          name: "Obstacle_BL",
        }, // Close the polygon
      ];

      setObstaclePolygon(obstaclePoints);

      const requestBody = {
        session_id: sessionID,
        obstacle_polygon: {
          points: obstaclePoints,
        },
      };

      const response = await getDStarLiteRoute(requestBody);

      if (response.d_star_lite_path.length > 0) {
        setDStarLitePath(response.d_star_lite_path.map((p) => [p.lat, p.lon]));
        toast({
          title: "Reroute Successful",
          description: "D* Lite path generated to avoid obstacle.",
        });
      } else {
        toast({
          title: "Reroute Failed",
          description:
            response.message || "Could not find a D* Lite reroute path.",
          variant: "destructive",
        });
      }
    } catch (error: any) {
      toast({
        title: "Error Rerouting",
        description: error.message || "Failed to generate D* Lite route.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    // Changed from isCalculating
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

  // If optimizedRouteData is null AND there are no unreachable destinations, then it's a full failure
  if (
    !optimizedRouteData ||
    (!optimizedRouteData.fullAstarPath.length &&
      !optimizedRouteData.unreachableDestinations?.length)
  ) {
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
    <div className="min-h-screen flex flex-col bg-background mt-[3dvh]">
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
        {/* {warnings.length > 0 && (
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
        )} */}

        {/* Main Content - All Visible Together */}
        <section className="py-12">
          <div className="container mx-auto px-4">
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
              {/* Metrics & Summary - Takes 1/3 width on large screens, placed first */}
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
                      <div className="p-2 rounded-lg bg-accent/10"></div>

                      <div className="flex items-center gap-3 p-3 rounded-lg bg-accent/5 border border-accent/10">
                        <div className="p-2 rounded-lg bg-accent/10">
                          <Clock className="h-5 w-5 text-accent" />
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">
                            Travel Time
                          </p>
                          <p className="text-xl font-bold">
                            {totalTime
                              ? `${totalTime.toFixed(1)} hrs`
                              : "-- hrs"}
                          </p>
                        </div>
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

                    {eta && (
                      <div className="flex items-center gap-3 p-3 rounded-lg bg-primary/5 border border-primary/10">
                        <div className="p-2 rounded-lg bg-primary/10">
                          <Clock className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">ETA</p>
                          <p className="text-xl font-bold">{eta}</p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Reroute with Obstacle Button */}
                <Button
                  onClick={handleGenerateObstacleAndReroute}
                  disabled={
                    isLoading || !optimizedRouteData?.fullAstarPath.length
                  }
                  className="w-full h-12 text-lg bg-gradient-to-r from-green-500 to-teal-600 hover:from-green-600 hover:to-teal-700 transition-all shadow-lg mt-6"
                >
                  Generate Obstacle & Reroute
                </Button>

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
                    routes={[
                      {
                        id: "astar",
                        start: startPoint,
                        end: portsForMap[portsForMap.length - 1],
                        path: optimizedRouteData.fullAstarPath,
                        distance: totalDistance,
                        fuelConsumption: totalFuel,
                        color: "red", // Original A* path in red
                      },
                      ...(dStarLitePath
                        ? [
                            {
                              id: "dstar",
                              start: currentLocationState || startPoint, // D* Lite starts from current location or startPoint
                              end: portsForMap[portsForMap.length - 1], // Ends at final destination
                              path: dStarLitePath,
                              color: "blue", // D* Lite path in blue
                            },
                          ]
                        : []),
                    ]}
                    currentLocation={currentLocationState}
                    obstaclePolygon={obstaclePolygon} // Pass obstacle polygon to map
                    onMapClick={handleMapClick} // Enable map clicks for D* Lite
                  />
                </CardContent>
                {/* Suggested DRL Actions */}
                <Card className="border-2 shadow-lg mt-8 xl:col-span-2 max-w-[95%] mx-auto">
                  <CardHeader>
                    <CardTitle className="text-lg">
                      Suggested DRL Actions
                    </CardTitle>
                    <CardDescription>
                      Next recommended speed and heading from the DRL agent.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center gap-3 p-3 rounded-lg bg-primary/5 border border-primary/10">
                      <div className="p-2 rounded-lg bg-primary/10">
                        <Ship className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">
                          Suggested Speed
                        </p>
                        <p className="text-xl font-bold">
                          {suggestedSpeed !== null
                            ? `${suggestedSpeed.toFixed(1)} knots`
                            : "-- knots"}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 p-3 rounded-lg bg-accent/5 border border-accent/10">
                      <div className="p-2 rounded-lg bg-accent/10">
                        <Navigation className="h-5 w-5 text-accent" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">
                          Suggested Heading
                        </p>
                        <p className="text-xl font-bold">
                          {suggestedHeading !== null
                            ? `${suggestedHeading.toFixed(1)}°`
                            : "-- °"}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </Card>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
};

export default RouteVisualization;
