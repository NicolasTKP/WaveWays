import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Ship, TrendingDown, Clock, Navigation, Anchor, AlertCircle, CheckCircle2, ArrowLeft } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { RouteMap } from "@/components/RouteMap";
import { Port, OptimizedRoute } from "@/types/route";
import { useToast } from "@/hooks/use-toast";
import MapView from "@/components/mapVisual";

const RouteVisualization = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [optimizedRoute, setOptimizedRoute] = useState<OptimizedRoute | null>(null);
  const [vesselConfig, setVesselConfig] = useState<any>(null);
  const [selectedPorts, setSelectedPorts] = useState<Port[]>([]);
  const [isCalculating, setIsCalculating] = useState(true);

  const [currentLocation, setCurrentLocation] = useState<Port | null>(null);

  const handleMapClick = (lat: number, lng: number) => {
    if (!vesselConfig || !optimizedRoute) return;

    const portName = `Custom Point ${optimizedRoute.ports.length + 1}`;
    const newPort: Port = {
      id: `custom-${Date.now()}`,       // unique id
      name: portName,
      lat,
      lng,
      country: "Unknown",
      coordinates: `${lat},${lng}`
    };

    const updatedPorts = [...optimizedRoute.ports, newPort];

    setSelectedPorts(updatedPorts);
    generateDemoRoute(vesselConfig, updatedPorts);
  };


  useEffect(() => {
    const storedSelectedPorts = sessionStorage.getItem('selectedPorts');
    if (storedSelectedPorts) {
      const ports: Port[] = JSON.parse(storedSelectedPorts);
      if (ports.length > 0) setCurrentLocation(ports[0]); // first port as current location
    }
  }, []);

  useEffect(() => {
    // Retrieve data from sessionStorage
    const storedVesselConfig = sessionStorage.getItem('vesselConfig');
    const storedSelectedPorts = sessionStorage.getItem('selectedPorts');

    if (!storedVesselConfig || !storedSelectedPorts) {
      toast({
        title: "No Route Data",
        description: "Please configure your route first.",
        variant: "destructive",
      });
      navigate('/plan-route');
      return;
    }

    const config = JSON.parse(storedVesselConfig);
    const ports = JSON.parse(storedSelectedPorts);

    setVesselConfig(config);
    setSelectedPorts(ports);

    // Generate demo route
    setTimeout(() => {
      generateDemoRoute(config, ports);
      setIsCalculating(false);
      toast({
        title: "Route Generated",
        description: `Optimized route with ${ports.length - 1} segments calculated successfully.`,
      });
    }, 1500);
  }, [navigate, toast]);

  const generateDemoRoute = (config: any, ports: Port[]) => {
    if (ports.length < 2) return;

    const segments = ports.slice(0, -1).map((port, index) => {
      const nextPort = ports[index + 1];
      const distance = calculateDistance(port, nextPort);
      const fuelConsumption = (distance / (config.speed * 24)) * config.fuelConsumption;
      const travelTime = distance / config.speed;

      return {
        start: port,
        end: nextPort,
        path: [[port.lat, port.lng] as [number, number], [nextPort.lat, nextPort.lng] as [number, number]],
        distance,
        fuelConsumption,
        travelTime,
      };
    });

    const totalDistance = segments.reduce((sum, seg) => sum + seg.distance, 0);
    const totalFuel = segments.reduce((sum, seg) => sum + seg.fuelConsumption, 0);
    const totalTime = segments.reduce((sum, seg) => sum + seg.travelTime, 0);

    setOptimizedRoute({
      segments,
      totalDistance,
      totalFuel,
      totalTime,
      ports,
      warnings: totalFuel > config.tankCapacity - config.safetyMargin
        ? ['⚠️ Fuel capacity may be insufficient for this route. Consider adding refueling stops.']
        : [],
    });
  };

  const calculateDistance = (port1: Port, port2: Port): number => {
    const R = 3440.065; // Earth radius in nautical miles
    const dLat = toRad(port2.lat - port1.lat);
    const dLon = toRad(port2.lng - port1.lng);
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(toRad(port1.lat)) * Math.cos(toRad(port2.lat)) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  const toRad = (deg: number) => deg * (Math.PI / 180);

  if (isCalculating) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <Ship className="h-16 w-16 text-primary animate-pulse mx-auto mb-4" />
          <h2 className="text-2xl font-bold mb-2">Calculating Optimal Route</h2>
          <p className="text-muted-foreground">Please wait while we optimize your maritime route...</p>
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
                onClick={() => navigate('/plan-route')}
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
              Route from <span className="font-semibold text-foreground">{selectedPorts[0]?.name}</span> to{' '}
              <span className="font-semibold text-foreground">{selectedPorts[selectedPorts.length - 1]?.name}</span>
            </p>
          </div>
        </section>

        {/* Warnings */}
        {optimizedRoute && optimizedRoute.warnings.length > 0 && (
          <section className="py-6 bg-destructive/5 border-b border-destructive/10">
            <div className="container mx-auto px-4">
              {optimizedRoute.warnings.map((warning, i) => (
                <div key={i} className="flex items-start gap-3 p-4 rounded-lg bg-background border border-destructive/20 max-w-4xl">
                  <AlertCircle className="h-5 w-5 text-destructive flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-destructive font-medium">{warning}</p>
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
                  <CardDescription>Interactive map showing your optimized route</CardDescription>
                </CardHeader>
                <CardContent>
                  <RouteMap
                    ports={optimizedRoute?.ports || [currentLocation]}
                    route={optimizedRoute?.segments || []}
                    currentLocation={currentLocation}
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
                        <p className="text-xs text-muted-foreground">Total Distance</p>
                        <p className="text-xl font-bold">
                          {optimizedRoute ? `${optimizedRoute.totalDistance.toFixed(1)} nm` : '-- nm'}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 p-3 rounded-lg bg-accent/5 border border-accent/10">
                      <div className="p-2 rounded-lg bg-accent/10">
                        <Clock className="h-5 w-5 text-accent" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Travel Time</p>
                        <p className="text-xl font-bold">
                          {optimizedRoute ? `${optimizedRoute.totalTime.toFixed(1)} hrs` : '-- hrs'}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 p-3 rounded-lg bg-muted">
                      <div className="p-2 rounded-lg bg-background">
                        <TrendingDown className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Fuel Usage</p>
                        <p className="text-xl font-bold">
                          {optimizedRoute ? `${optimizedRoute.totalFuel.toFixed(1)} tons` : '-- tons'}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 p-3 rounded-lg bg-muted">
                      <div className="p-2 rounded-lg bg-background">
                        <Anchor className="h-5 w-5 text-muted-foreground" />
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Total Ports</p>
                        <p className="text-xl font-bold">
                          {optimizedRoute ? optimizedRoute.ports.length : '--'}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Route Summary */}
                {optimizedRoute && (
                  <Card className="border-2 shadow-lg">
                    <CardHeader>
                      <CardTitle className="text-lg">Route Summary</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-muted-foreground">Origin</p>
                        <div className="p-3 rounded-lg bg-accent/5 border border-accent/10">
                          <p className="font-semibold">{optimizedRoute.ports[0]?.name}</p>
                          <p className="text-xs text-muted-foreground">{optimizedRoute.ports[0]?.country}</p>
                        </div>
                      </div>

                      {optimizedRoute.ports.length > 2 && (
                        <div className="space-y-2">
                          <p className="text-sm font-medium text-muted-foreground">Waypoints</p>
                          <div className="space-y-2">
                            {optimizedRoute.ports.slice(1, -1).map((port, idx) => (
                              <div key={idx} className="p-2 rounded-lg bg-muted text-sm">
                                <p className="font-medium">{port.name}</p>
                                <p className="text-xs text-muted-foreground">{port.country}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="space-y-2">
                        <p className="text-sm font-medium text-muted-foreground">Destination</p>
                        <div className="p-3 rounded-lg bg-primary/5 border border-primary/10">
                          <p className="font-semibold">{optimizedRoute.ports[optimizedRoute.ports.length - 1]?.name}</p>
                          <p className="text-xs text-muted-foreground">{optimizedRoute.ports[optimizedRoute.ports.length - 1]?.country}</p>
                        </div>
                      </div>

                      <div className="pt-4 border-t space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Route Segments</span>
                          <span className="font-semibold">{optimizedRoute.segments.length}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Avg Speed</span>
                          <span className="font-semibold">{vesselConfig?.speed} knots</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Fuel Efficiency</span>
                          <span className="font-semibold">
                            {((optimizedRoute.totalFuel / optimizedRoute.totalDistance) * 100).toFixed(2)} tons/100nm
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
