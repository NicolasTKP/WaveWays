import { Ship, Waves, TrendingDown, MapPin, Cloud, Shield, BarChart, Compass } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export const Features = () => {
  const features = [
    {
      icon: Compass,
      title: "Advanced Pathfinding",
      description: "Utilizing A* and D* Lite algorithms for optimal route calculation considering dynamic obstacles and conditions",
    },
    {
      icon: Waves,
      title: "Weather Integration",
      description: "Real-time marine forecasts including wave heights, wind speeds, and sea conditions for safer voyages",
    },
    {
      icon: TrendingDown,
      title: "Fuel Efficiency",
      description: "Minimize fuel consumption with intelligent route planning that considers vessel specifications and cargo weight",
    },
    {
      icon: MapPin,
      title: "Multi-Port Planning",
      description: "Optimize routes across multiple ports with traveling salesman problem (TSP) algorithms",
    },
    {
      icon: Cloud,
      title: "Environmental Data",
      description: "Factor in bathymetry, EEZ zones, and environmental conditions for compliant and safe navigation",
    },
    {
      icon: Shield,
      title: "Safety Margins",
      description: "Built-in safety fuel margins and draft considerations to ensure vessel security throughout the journey",
    },
    {
      icon: BarChart,
      title: "Cost Analysis",
      description: "Comprehensive cost breakdowns including fuel consumption, time estimates, and distance metrics",
    },
    {
      icon: Ship,
      title: "Vessel Profiles",
      description: "Customizable vessel parameters including speed, capacity, dimensions, and fuel consumption rates",
    },
  ];

  return (
    <section id="features" className="py-24 bg-background">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary mb-4">
            <Ship className="h-4 w-4 text-secondary-foreground" />
            <span className="text-sm font-medium text-secondary-foreground">
              Powerful Features
            </span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Everything You Need for Optimal Routing
          </h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Advanced algorithms and real-time data combined to deliver the most efficient maritime routes
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <Card
              key={index}
              className="group hover:shadow-lg hover:scale-105 transition-all duration-300 border-2 hover:border-accent/50 animate-fade-in-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <CardContent className="p-6">
                <div className="mb-4 inline-flex p-3 rounded-xl bg-primary/10 text-primary group-hover:bg-accent group-hover:text-accent-foreground transition-colors">
                  <feature.icon className="h-6 w-6" />
                </div>
                <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {feature.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};
