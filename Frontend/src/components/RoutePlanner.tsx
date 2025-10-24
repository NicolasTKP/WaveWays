import { useState } from "react";
import { Ship, Navigation, TrendingDown, Clock, Anchor, Waves, AlertCircle, CheckCircle2, MapPin, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { RouteMap } from "./RouteMap";
import { PortSelector } from "./PortSelector";
import { Port, OptimizedRoute } from "@/types/route";
import { useToast } from "@/hooks/use-toast";

export const RoutePlanner = () => {
  const { toast } = useToast();
  const [vesselConfig, setVesselConfig] = useState({
    speed: 20.0,
    fuelConsumption: 20.0,
    tankCapacity: 1000.0,
    safetyMargin: 10.0,
    length: 1000,
    width: 500,
    height: 60,
    underwaterPercent: 30,
  });

  const [currentLocation] = useState<Port>({
    id: 'current',
    name: 'Current Vessel Location',
    lat: 1.3521,
    lng: 103.8198,
    country: 'Singapore',
    coordinates: '1.3521, 103.8198'
  });

  const [selectedPorts, setSelectedPorts] = useState<Port[]>([]);
  const [optimizedRoute, setOptimizedRoute] = useState<OptimizedRoute | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  const [isRoutePlannerOpen, setIsRoutePlannerOpen] = useState(true);
  const [isVesselConfigOpen, setIsVesselConfigOpen] = useState(true);

  const handleCalculateRoute = () => {
    if (selectedPorts.length < 2) {
      toast({
        title: "Insufficient Ports",
        description: "Please select at least 2 ports (start and end).",
        variant: "destructive",
      });
      return;
    }

    setIsCalculating(true);

    // Scroll to top of route planner section smoothly
    const routePlannerSection = document.getElementById("route-planner");
    if (routePlannerSection) {
      routePlannerSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    
    // Using demo mode for frontend-only version
    setTimeout(() => {
      generateDemoRoute();
      setIsCalculating(false);
      toast({
        title: "Route Calculated",
        description: `Optimized route with ${selectedPorts.length - 1} segments generated successfully.`,
      });
    }, 1000);
  };

  const generateDemoRoute = () => {
    if (selectedPorts.length < 2) return;

    const segments = selectedPorts.slice(0, -1).map((port, index) => {
      const nextPort = selectedPorts[index + 1];
      const distance = calculateDistance(port, nextPort);
      const fuelConsumption = (distance / (vesselConfig.speed * 24)) * vesselConfig.fuelConsumption;
      const travelTime = distance / vesselConfig.speed;

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
      ports: selectedPorts,
      warnings: totalFuel > vesselConfig.tankCapacity - vesselConfig.safetyMargin 
        ? ['⚠️ Fuel capacity may be insufficient for this route']
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

  return (
    <section id="route-planner" className="py-24 bg-muted/30 scroll-mt-20">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 mb-4">
            <Navigation className="h-4 w-4 text-accent" />
            <span className="text-sm font-medium text-accent">
              Route Planning
            </span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            Plan Your Optimal RouteAAA
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Configure your vessel parameters and generate optimized maritime routes
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Split into Route Planner and Vessel Configuration */}
          <div className="lg:col-span-1 space-y-4">
            {/* Route Planner Section */}
            <Card className="border-2">
              <Collapsible open={isRoutePlannerOpen} onOpenChange={setIsRoutePlannerOpen}>
                <CardHeader className="pb-3">
                  <CollapsibleTrigger asChild>
                    <div className="flex items-center justify-between cursor-pointer group">
                      <div className="flex items-center gap-2">
                        <MapPin className="h-5 w-5 text-primary" />
                        <CardTitle className="text-lg">Route Planner</CardTitle>
                      </div>
                      {isRoutePlannerOpen ? (
                        <ChevronUp className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                      ) : (
                        <ChevronDown className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                      )}
                    </div>
                  </CollapsibleTrigger>
                  <CardDescription>Select start and destination ports</CardDescription>
                </CardHeader>
                <CollapsibleContent>
                  <CardContent>
                    <PortSelector 
                      selectedPorts={selectedPorts}
                      onPortsChange={setSelectedPorts}
                    />
                  </CardContent>
                </CollapsibleContent>
              </Collapsible>
            </Card>

            {/* Vessel Configuration Section */}
            <Card className="border-2">
              <Collapsible open={isVesselConfigOpen} onOpenChange={setIsVesselConfigOpen}>
                <CardHeader className="pb-3">
                  <CollapsibleTrigger asChild>
                    <div className="flex items-center justify-between cursor-pointer group">
                      <div className="flex items-center gap-2">
                        <Ship className="h-5 w-5 text-primary" />
                        <CardTitle className="text-lg">Vessel Configuration</CardTitle>
                      </div>
                      {isVesselConfigOpen ? (
                        <ChevronUp className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                      ) : (
                        <ChevronDown className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                      )}
                    </div>
                  </CollapsibleTrigger>
                  <CardDescription>Set your vessel specifications</CardDescription>
                </CardHeader>
                <CollapsibleContent>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="speed">Speed (knots)</Label>
                      <Input
                        id="speed"
                        type="number"
                        value={vesselConfig.speed}
                        onChange={(e) => setVesselConfig({ ...vesselConfig, speed: parseFloat(e.target.value) || 0 })}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="fuel">Fuel Consumption (tons/day)</Label>
                      <Input
                        id="fuel"
                        type="number"
                        value={vesselConfig.fuelConsumption}
                        onChange={(e) => setVesselConfig({ ...vesselConfig, fuelConsumption: parseFloat(e.target.value) || 0 })}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="capacity">Tank Capacity (tons)</Label>
                      <Input
                        id="capacity"
                        type="number"
                        value={vesselConfig.tankCapacity}
                        onChange={(e) => setVesselConfig({ ...vesselConfig, tankCapacity: parseFloat(e.target.value) || 0 })}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="margin">Safety Margin (tons)</Label>
                      <Input
                        id="margin"
                        type="number"
                        value={vesselConfig.safetyMargin}
                        onChange={(e) => setVesselConfig({ ...vesselConfig, safetyMargin: parseFloat(e.target.value) || 0 })}
                      />
                    </div>

                    <div className="grid grid-cols-3 gap-2">
                      <div className="space-y-2">
                        <Label htmlFor="length">Length (m)</Label>
                        <Input
                          id="length"
                          type="number"
                          value={vesselConfig.length}
                          onChange={(e) => setVesselConfig({ ...vesselConfig, length: parseFloat(e.target.value) || 0 })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="width">Width (m)</Label>
                        <Input
                          id="width"
                          type="number"
                          value={vesselConfig.width}
                          onChange={(e) => setVesselConfig({ ...vesselConfig, width: parseFloat(e.target.value) || 0 })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="height">Height (m)</Label>
                        <Input
                          id="height"
                          type="number"
                          value={vesselConfig.height}
                          onChange={(e) => setVesselConfig({ ...vesselConfig, height: parseFloat(e.target.value) || 0 })}
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="underwater">Underwater (%)</Label>
                      <Input
                        id="underwater"
                        type="number"
                        value={vesselConfig.underwaterPercent}
                        onChange={(e) => setVesselConfig({ ...vesselConfig, underwaterPercent: parseFloat(e.target.value) || 0 })}
                      />
                    </div>
                  </CardContent>
                </CollapsibleContent>
              </Collapsible>
            </Card>

            {/* Calculate Route Button */}
            <Button 
              className="w-full bg-primary hover:bg-primary/90 h-12"
              onClick={handleCalculateRoute}
              disabled={isCalculating || selectedPorts.length < 2}
            >
              {isCalculating ? "Calculating Route..." : "Calculate Route"}
            </Button>
            
            {optimizedRoute && optimizedRoute.warnings.length > 0 && (
              <div className="space-y-2">
                {optimizedRoute.warnings.map((warning, i) => (
                  <div key={i} className="flex items-start gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/20">
                    <AlertCircle className="h-4 w-4 text-destructive flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-destructive">{warning}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Right Panel - Route Visualization & Results */}
          <Card className="lg:col-span-2 border-2">
            <CardHeader>
              <CardTitle>Route Visualization</CardTitle>
              <CardDescription>
                {optimizedRoute 
                  ? "View your optimized route and analytics" 
                  : "Configure vessel and select ports to generate route"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="map" className="w-full">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="map">Map View</TabsTrigger>
                  <TabsTrigger value="metrics">Metrics</TabsTrigger>
                  <TabsTrigger value="summary">Summary</TabsTrigger>
                </TabsList>

                <TabsContent value="map" className="space-y-4 mt-6">
                  <RouteMap
                    ports={optimizedRoute?.ports || [currentLocation]}
                    route={optimizedRoute?.segments || []}
                    currentLocation={currentLocation}
                  />
                </TabsContent>

                <TabsContent value="metrics" className="space-y-4 mt-6">
                  <div className="grid grid-cols-2 gap-4">
                    <Card className="bg-card/50">
                      <CardContent className="pt-6">
                        <div className="flex items-center gap-3">
                          <div className="p-3 rounded-lg bg-primary/10">
                            <TrendingDown className="h-5 w-5 text-primary" />
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Fuel Usage</p>
                            <p className="text-2xl font-bold">
                              {optimizedRoute ? `${optimizedRoute.totalFuel.toFixed(1)} tons` : '-- tons'}
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-card/50">
                      <CardContent className="pt-6">
                        <div className="flex items-center gap-3">
                          <div className="p-3 rounded-lg bg-accent/10">
                            <Clock className="h-5 w-5 text-accent" />
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Travel Time</p>
                            <p className="text-2xl font-bold">
                              {optimizedRoute ? `${optimizedRoute.totalTime.toFixed(1)} hours` : '-- hours'}
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-card/50">
                      <CardContent className="pt-6">
                        <div className="flex items-center gap-3">
                          <div className="p-3 rounded-lg bg-secondary/50">
                            <Navigation className="h-5 w-5 text-secondary-foreground" />
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Distance</p>
                            <p className="text-2xl font-bold">
                              {optimizedRoute ? `${optimizedRoute.totalDistance.toFixed(1)} nm` : '-- nm'}
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-card/50">
                      <CardContent className="pt-6">
                        <div className="flex items-center gap-3">
                          <div className="p-3 rounded-lg bg-muted">
                            <Anchor className="h-5 w-5 text-muted-foreground" />
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Ports</p>
                            <p className="text-2xl font-bold">
                              {optimizedRoute ? optimizedRoute.ports.length : '--'}
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                  
                  {optimizedRoute && (
                    <div className="mt-6 p-4 rounded-lg border bg-card/30 space-y-3">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="h-5 w-5 text-accent" />
                        <h4 className="font-semibold">Route Summary</h4>
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Segments:</span>{' '}
                          <span className="font-medium">{optimizedRoute.segments.length}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Avg Speed:</span>{' '}
                          <span className="font-medium">{vesselConfig.speed} knots</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Fuel Remaining:</span>{' '}
                          <span className="font-medium">
                            {(vesselConfig.tankCapacity - optimizedRoute.totalFuel).toFixed(1)} tons
                          </span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Safety Margin:</span>{' '}
                          <span className={`font-medium ${
                            (vesselConfig.tankCapacity - optimizedRoute.totalFuel) > vesselConfig.safetyMargin
                              ? 'text-accent'
                              : 'text-destructive'
                          }`}>
                            {((vesselConfig.tankCapacity - optimizedRoute.totalFuel) > vesselConfig.safetyMargin ? '✓' : '✗')} 
                            {' '}{vesselConfig.safetyMargin} tons
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="summary" className="space-y-4 mt-6">
                  {optimizedRoute && optimizedRoute.segments.length > 0 ? (
                    <div className="space-y-3">
                      {optimizedRoute.segments.map((segment, index) => (
                        <div
                          key={index}
                          className="p-4 rounded-lg border bg-card/50 hover:bg-card transition-colors"
                        >
                          <div className="flex items-start justify-between mb-3">
                            <div className="flex items-center gap-3">
                              <Badge variant="outline" className="bg-accent/10 text-accent border-accent/20">
                                Leg {index + 1}
                              </Badge>
                              <div>
                                <p className="font-semibold text-sm">
                                  {segment.start.name} → {segment.end.name}
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  {segment.start.country} to {segment.end.country}
                                </p>
                              </div>
                            </div>
                          </div>
                          <div className="grid grid-cols-3 gap-4 text-sm">
                            <div className="flex items-center gap-2">
                              <Navigation className="h-4 w-4 text-muted-foreground" />
                              <div>
                                <p className="text-xs text-muted-foreground">Distance</p>
                                <p className="font-medium">{segment.distance.toFixed(1)} nm</p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <TrendingDown className="h-4 w-4 text-muted-foreground" />
                              <div>
                                <p className="text-xs text-muted-foreground">Fuel</p>
                                <p className="font-medium">{segment.fuelConsumption.toFixed(1)} tons</p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <Clock className="h-4 w-4 text-muted-foreground" />
                              <div>
                                <p className="text-xs text-muted-foreground">Time</p>
                                <p className="font-medium">{segment.travelTime.toFixed(1)} hrs</p>
                              </div>
                            </div>
                          </div>
                          {segment.weatherConditions && (
                            <div className="mt-3 pt-3 border-t flex items-center gap-2 text-xs text-muted-foreground">
                              <Waves className="h-3 w-3" />
                              <span>Wave: {segment.weatherConditions.waveHeight}m</span>
                              <span>•</span>
                              <span>Wind: {segment.weatherConditions.windSpeed} knots</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-12">
                      <Anchor className="h-12 w-12 mx-auto mb-3 text-muted-foreground opacity-50" />
                      <p className="text-muted-foreground">
                        Calculate route to view summary details
                      </p>
                      <div className="mt-6 p-4 rounded-lg border bg-card/30 text-left max-w-md mx-auto">
                        <div className="flex items-center gap-2 mb-3">
                          <Ship className="h-5 w-5 text-primary" />
                          <h4 className="font-semibold">Current Vessel Status</h4>
                        </div>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Location:</span>
                            <span className="font-medium">{currentLocation.name}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Speed:</span>
                            <span className="font-medium">{vesselConfig.speed} knots</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Fuel Capacity:</span>
                            <span className="font-medium">{vesselConfig.tankCapacity} tons</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
};
