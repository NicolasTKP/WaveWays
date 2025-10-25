import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Ship,
  Navigation,
  MapPin,
  ChevronDown,
  ChevronUp,
  List,
  Map,
  Loader2,
  AlertCircle,
  CheckCircle2,
  ArrowUp,
  ArrowDown,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PortSelector } from "@/components/PortSelector";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { Port } from "@/types/route";
import { useToast } from "@/hooks/use-toast";
import { RouteMap } from "@/components/RouteMap";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  suggestRouteSequence,
  SuggestedRouteSequenceResponse,
  PointModel,
  UnreachableDestination,
} from "@/services/routeService";
const PlanRoute = () => {
  const navigate = useNavigate();
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

  const [selectedPorts, setSelectedPorts] = useState<Port[]>([]);
  const [isRoutePlannerOpen, setIsRoutePlannerOpen] = useState(true);
  const [isVesselConfigOpen, setIsVesselConfigOpen] = useState(true);
  const [selectionMode, setSelectionMode] = useState<"list" | "map">("list");

  const [sessionID, setSessionID] = useState<string | null>(null);
  const [suggestedSequence, setSuggestedSequence] = useState<PointModel[]>([]);
  const [unreachableDestinations, setUnreachableDestinations] = useState<
    UnreachableDestination[]
  >([]);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Default current location (Singapore Strait)
  const currentLocation = { lat: 1.2921, lng: 103.8198 };

  const handleGenerateRoute = async () => {
    if (selectedPorts.length < 2) {
      toast({
        title: "Insufficient Ports",
        description: "Please select at least 2 ports (start and destination).",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      const startPoint = selectedPorts[0];
      const destinations = selectedPorts.slice(1);

      const requestBody = {
        start_point: {
          lat: startPoint.lat,
          lon: startPoint.lng,
          name: startPoint.name,
        },
        destinations: destinations.map((p) => ({
          lat: p.lat,
          lon: p.lng,
          name: p.name,
        })),
        is_cycle_route: false, // Assuming non-cycle route for now
        vessel_config: {
          speed_knots: vesselConfig.speed,
          fuel_consumption_t_per_day: vesselConfig.fuelConsumption,
          fuel_tank_capacity_t: vesselConfig.tankCapacity,
          safety_fuel_margin_t: vesselConfig.safetyMargin,
          length: vesselConfig.length,
          width: vesselConfig.width,
          height: vesselConfig.height,
          underwater_percent: vesselConfig.underwaterPercent,
        },
      };

      const response = await suggestRouteSequence(requestBody);
      setSessionID(response.session_id);
      setSuggestedSequence(response.suggested_sequence);
      setUnreachableDestinations(response.unreachable_destinations);
      setIsDialogOpen(true); // Open the dialog to show the sequence

      toast({
        title: "Route Sequence Suggested",
        description: "Review the suggested sequence and unreachable ports.",
      });
    } catch (error: any) {
      toast({
        title: "Error Generating Route",
        description: error.message || "Failed to get suggested route sequence.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleMoveUp = (index: number) => {
    if (index === 0) return; // Cannot move the first item up

    setSuggestedSequence((prevSequence) => {
      const newSequence = [...prevSequence];
      const [movedItem] = newSequence.splice(index, 1);
      newSequence.splice(index - 1, 0, movedItem);
      return newSequence;
    });
  };

  const handleMoveDown = (index: number) => {
    if (index === suggestedSequence.length - 1) return; // Cannot move the last item down

    setSuggestedSequence((prevSequence) => {
      const newSequence = [...prevSequence];
      const [movedItem] = newSequence.splice(index, 1);
      newSequence.splice(index + 1, 0, movedItem);
      return newSequence;
    });
  };

  const handleConfirmSequenceAndNavigate = () => {
    if (!sessionID) {
      toast({
        title: "Error",
        description: "Session ID is missing. Please regenerate the route.",
        variant: "destructive",
      });
      return;
    }

    // Pass data via navigate state
    navigate("/route-visualization", {
      state: {
        sessionID,
        vesselConfig: {
          speed_knots: vesselConfig.speed,
          fuel_consumption_t_per_day: vesselConfig.fuelConsumption,
          fuel_tank_capacity_t: vesselConfig.tankCapacity,
          safety_fuel_margin_t: vesselConfig.safetyMargin,
          length: vesselConfig.length,
          width: vesselConfig.width,
          height: vesselConfig.height,
          underwater_percent: vesselConfig.underwaterPercent,
        },
        sequencedDestinations: suggestedSequence,
        startPoint: selectedPorts[0], // The original start point
      },
    });
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Navbar />

      <main className="flex-1">
        {/* Hero Section */}
        <section className="relative py-24 bg-gradient-to-b from-primary/5 via-accent/5 to-background overflow-hidden">
          <div className="absolute inset-0 bg-[url('/ocean-pattern.jpg')] opacity-5 bg-cover bg-center"></div>
          <div className="container mx-auto px-4 relative z-10">
            <div className="text-center mb-12 animate-fade-in">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 mb-6">
                <Navigation className="h-5 w-5 text-accent" />
                <span className="text-sm font-medium text-accent">
                  Maritime Route Planning
                </span>
              </div>
              <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-primary via-accent to-primary">
                Plan Your Optimal Route
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
                Configure your vessel parameters and select ports to generate an
                optimized maritime route
              </p>
            </div>
          </div>
        </section>

        {/* Map Section */}
        <section className="container mx-auto px-4 -mt-16 z-10 relative">
          <RouteMap
            currentLocation={{
              lat: currentLocation.lat,
              lng: currentLocation.lng,
              name: "You Are Here",
            }}
            ports={selectedPorts}
            zoom={12}
          />
        </section>

        {/* Configuration Section */}
        <section className="py-16 bg-muted/30">
          <div className="container mx-auto px-4">
            <div className="max-w-[60dvw] mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Route Planner Card */}
              <Card className="border-2 shadow-lg">
                <Collapsible
                  open={isRoutePlannerOpen}
                  onOpenChange={setIsRoutePlannerOpen}
                >
                  <CardHeader className="pb-4">
                    <CollapsibleTrigger asChild>
                      <div className="flex items-center justify-between cursor-pointer group">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg bg-primary/10">
                            <MapPin className="h-6 w-6 text-primary" />
                          </div>
                          <div>
                            <CardTitle className="text-xl">
                              Route Planner
                            </CardTitle>
                            <CardDescription className="text-sm mt-1">
                              Select start and destination ports
                            </CardDescription>
                          </div>
                        </div>
                        {isRoutePlannerOpen ? (
                          <ChevronUp className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                        ) : (
                          <ChevronDown className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                        )}
                      </div>
                    </CollapsibleTrigger>
                  </CardHeader>
                  <CollapsibleContent>
                    <CardContent className="pt-0">
                      <Tabs
                        value={selectionMode}
                        onValueChange={(v) =>
                          setSelectionMode(v as "list" | "map")
                        }
                        className="w-full"
                      >
                        <TabsList className="grid w-full grid-cols-2 mb-4">
                          <TabsTrigger
                            value="list"
                            className="flex items-center gap-2"
                          >
                            <List className="h-4 w-4" />
                            Select from List
                          </TabsTrigger>
                          <TabsTrigger
                            value="map"
                            className="flex items-center gap-2"
                          >
                            <Map className="h-4 w-4" />
                            Click on Map
                          </TabsTrigger>
                        </TabsList>

                        <TabsContent value="list" className="mt-0">
                          <PortSelector
                            selectedPorts={selectedPorts}
                            onPortsChange={setSelectedPorts}
                          />
                        </TabsContent>

                        <TabsContent value="map" className="mt-0">
                          <div className="rounded-lg overflow-hidden border h-[400px]">
                            <RouteMap
                              center={currentLocation}
                              ports={selectedPorts}
                              zoom={6}
                              onMapClick={(lat, lng) => {
                                const portName = `Custom Point ${
                                  selectedPorts.length + 1
                                }`;
                                const newPort: Port = {
                                  id: `custom-${Date.now()}`,
                                  name: portName,
                                  lat,
                                  lng,
                                  country: "Unknown",
                                  coordinates: `${lat},${lng}`,
                                };
                                setSelectedPorts((prevPorts) => [
                                  ...prevPorts,
                                  newPort,
                                ]);
                              }}
                            />
                          </div>
                        </TabsContent>
                      </Tabs>

                      <div className="mt-4 p-3 rounded-lg bg-accent/5 border border-accent/10">
                        <p className="text-sm text-muted-foreground">
                          <span className="font-semibold text-foreground">
                            {selectedPorts.length}
                          </span>{" "}
                          port{selectedPorts.length !== 1 ? "s" : ""} selected
                        </p>
                      </div>
                    </CardContent>
                  </CollapsibleContent>
                </Collapsible>
              </Card>

              {/* Vessel Configuration Card */}
              <Card className="border-2 shadow-lg">
                <Collapsible
                  open={isVesselConfigOpen}
                  onOpenChange={setIsVesselConfigOpen}
                >
                  <CardHeader className="pb-4">
                    <CollapsibleTrigger asChild>
                      <div className="flex items-center justify-between cursor-pointer group">
                        <div className="flex items-center gap-3">
                          <div className="p-2 rounded-lg bg-primary/10">
                            <Ship className="h-6 w-6 text-primary" />
                          </div>
                          <div>
                            <CardTitle className="text-xl">
                              Vessel Configuration
                            </CardTitle>
                            <CardDescription className="text-sm mt-1">
                              Set your vessel specifications
                            </CardDescription>
                          </div>
                        </div>
                        {isVesselConfigOpen ? (
                          <ChevronUp className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                        ) : (
                          <ChevronDown className="h-5 w-5 text-muted-foreground group-hover:text-foreground transition-colors" />
                        )}
                      </div>
                    </CollapsibleTrigger>
                  </CardHeader>
                  <CollapsibleContent>
                    <CardContent className="pt-0 space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label
                            htmlFor="speed"
                            className="text-sm font-medium"
                          >
                            Speed (knots)
                          </Label>
                          <Input
                            id="speed"
                            type="number"
                            value={vesselConfig.speed}
                            onChange={(e) =>
                              setVesselConfig({
                                ...vesselConfig,
                                speed: parseFloat(e.target.value) || 0,
                              })
                            }
                            className="h-10"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="fuel" className="text-sm font-medium">
                            Fuel (tons/day)
                          </Label>
                          <Input
                            id="fuel"
                            type="number"
                            value={vesselConfig.fuelConsumption}
                            onChange={(e) =>
                              setVesselConfig({
                                ...vesselConfig,
                                fuelConsumption:
                                  parseFloat(e.target.value) || 0,
                              })
                            }
                            className="h-10"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label
                            htmlFor="capacity"
                            className="text-sm font-medium"
                          >
                            Tank Capacity (tons)
                          </Label>
                          <Input
                            id="capacity"
                            type="number"
                            value={vesselConfig.tankCapacity}
                            onChange={(e) =>
                              setVesselConfig({
                                ...vesselConfig,
                                tankCapacity: parseFloat(e.target.value) || 0,
                              })
                            }
                            className="h-10"
                          />
                        </div>

                        <div className="space-y-2">
                          <Label
                            htmlFor="margin"
                            className="text-sm font-medium"
                          >
                            Safety Margin (tons)
                          </Label>
                          <Input
                            id="margin"
                            type="number"
                            value={vesselConfig.safetyMargin}
                            onChange={(e) =>
                              setVesselConfig({
                                ...vesselConfig,
                                safetyMargin: parseFloat(e.target.value) || 0,
                              })
                            }
                            className="h-10"
                          />
                        </div>
                      </div>

                      <div className="pt-2">
                        <Label className="text-sm font-medium mb-3 block">
                          Vessel Dimensions
                        </Label>
                        <div className="grid grid-cols-3 gap-3">
                          <div className="space-y-2">
                            <Label
                              htmlFor="length"
                              className="text-xs text-muted-foreground"
                            >
                              Length (m)
                            </Label>
                            <Input
                              id="length"
                              type="number"
                              value={vesselConfig.length}
                              onChange={(e) =>
                                setVesselConfig({
                                  ...vesselConfig,
                                  length: parseFloat(e.target.value) || 0,
                                })
                              }
                              className="h-9"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label
                              htmlFor="width"
                              className="text-xs text-muted-foreground"
                            >
                              Width (m)
                            </Label>
                            <Input
                              id="width"
                              type="number"
                              value={vesselConfig.width}
                              onChange={(e) =>
                                setVesselConfig({
                                  ...vesselConfig,
                                  width: parseFloat(e.target.value) || 0,
                                })
                              }
                              className="h-9"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label
                              htmlFor="height"
                              className="text-xs text-muted-foreground"
                            >
                              Height (m)
                            </Label>
                            <Input
                              id="height"
                              type="number"
                              value={vesselConfig.height}
                              onChange={(e) =>
                                setVesselConfig({
                                  ...vesselConfig,
                                  height: parseFloat(e.target.value) || 0,
                                })
                              }
                              className="h-9"
                            />
                          </div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label
                          htmlFor="underwater"
                          className="text-sm font-medium"
                        >
                          Underwater (%)
                        </Label>
                        <Input
                          id="underwater"
                          type="number"
                          value={vesselConfig.underwaterPercent}
                          onChange={(e) =>
                            setVesselConfig({
                              ...vesselConfig,
                              underwaterPercent:
                                parseFloat(e.target.value) || 0,
                            })
                          }
                          className="h-10"
                        />
                      </div>
                    </CardContent>
                  </CollapsibleContent>
                </Collapsible>
              </Card>
            </div>

            {/* Generate Route Button */}
            <div className="max-w-5xl mx-auto mt-12">
              <Button
                size="lg"
                className="w-full h-14 text-lg bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 transition-all shadow-lg"
                onClick={handleGenerateRoute}
                disabled={selectedPorts.length < 2 || isLoading}
              >
                {isLoading ? (
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                ) : (
                  <Navigation className="mr-2 h-5 w-5" />
                )}
                {isLoading ? "Generating Route..." : "Generate Optimized Route"}
              </Button>
              {selectedPorts.length < 2 && (
                <p className="text-center text-sm text-muted-foreground mt-3">
                  Please select at least 2 ports to generate a route
                </p>
              )}
            </div>
          </div>
        </section>
      </main>

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Navigation className="h-6 w-6 text-primary" />
              Suggested Route Sequence
            </DialogTitle>
            <DialogDescription>
              Review the suggested sequence of destinations. You can drag and
              drop to reorder them.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            {suggestedSequence.length > 0 && (
              <ul className="space-y-2">
                {suggestedSequence.map((point, index) => (
                  <li
                    key={`${point.lat}-${point.lon}-${index}`}
                    className="flex items-center gap-3 p-3 border rounded-lg bg-card shadow-sm"
                  >
                    <span className="font-bold text-primary">{index + 1}.</span>
                    <MapPin className="h-5 w-5 text-muted-foreground" />
                    <span>
                      {point.name
                        ? `${index === 0 ? "Departure" : "Destination"}: ${
                            point.name
                          }`
                        : `${
                            index === 0 ? "Departure" : "Destination"
                          }: Lat ${point.lat.toFixed(
                            4
                          )}, Lon ${point.lon.toFixed(4)}`}
                    </span>
                    <div className="ml-auto flex gap-2">
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => handleMoveUp(index)}
                        disabled={index === 0}
                      >
                        <ArrowUp className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="icon"
                        onClick={() => handleMoveDown(index)}
                        disabled={index === suggestedSequence.length - 1}
                      >
                        <ArrowDown className="h-4 w-4" />
                      </Button>
                    </div>
                  </li>
                ))}
              </ul>
            )}

            {unreachableDestinations.length > 0 && (
              <div className="space-y-2 p-4 bg-destructive/5 border border-destructive/20 rounded-lg">
                <h3 className="text-lg font-semibold text-destructive flex items-center gap-2">
                  <AlertCircle className="h-5 w-5" /> Unreachable Destinations
                </h3>
                <ul className="list-disc pl-5 text-sm text-destructive-foreground">
                  {unreachableDestinations.map((dest, index) => (
                    <li key={index}>
                      Lat {dest.lat.toFixed(4)}, Lon {dest.lon.toFixed(4)}:{" "}
                      {dest.reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsDialogOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleConfirmSequenceAndNavigate}>
              Confirm Sequence & View Route
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Footer />
    </div>
  );
};

export default PlanRoute;
