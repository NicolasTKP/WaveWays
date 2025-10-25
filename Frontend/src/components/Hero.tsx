import { ArrowRight, Ship, TrendingDown, Map, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import heroShip from "@/assets/hero-ship.jpg";

export const Hero = () => {
  const navigate = useNavigate();
  
  return (
    <section id="home" className="relative min-h-screen flex items-center overflow-hidden pt-16">
      {/* Background Image with Overlay */}
      <div className="absolute inset-0 z-0">
        <img
          src={heroShip}
          alt="Cargo ship at sea"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-br from-primary/90 via-primary/80 to-accent/70" />
      </div>

      {/* Animated Background Elements */}
      <div className="absolute inset-0 z-0">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-accent/20 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-1/3 right-1/4 w-96 h-96 bg-primary/30 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }} />
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 z-10 relative">
        <div className="max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-background/10 backdrop-blur-sm border border-background/20 mb-6 animate-fade-in">
            <Ship className="h-4 w-4 text-primary-foreground" />
            <span className="text-sm font-medium text-primary-foreground">
              Vessel Route Optimization System
            </span>
          </div>

          <h1 className="text-5xl md:text-7xl font-bold text-primary-foreground mb-6 animate-fade-in-up">
            Navigate Smarter with{" "}
            <span className="bg-gradient-to-r from-accent to-background bg-clip-text text-transparent">
              WaveWays
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-primary-foreground/90 mb-8 max-w-2xl mx-auto animate-fade-in-up" style={{ animationDelay: "0.1s" }}>
            Optimize maritime routes with AI-powered path planning, real-time weather data, and fuel efficiency analytics
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12 animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
            <Button 
              size="lg" 
              className="bg-background text-foreground hover:bg-background/90 text-lg px-8 py-6"
              onClick={() => navigate('/plan-route')}
            >
              Start Planning
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
            <Button 
              size="lg" 
              variant="outline" 
              className="border-2 border-background text-background hover:bg-background hover:text-foreground text-lg px-8 py-6"
              onClick={() => navigate('/plan-route')}
            >
              View Demo
            </Button>
          </div>

          {/* Key Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto animate-fade-in-up" style={{ animationDelay: "0.3s" }}>
            <div className="flex flex-col items-center gap-3 p-6 rounded-2xl bg-background/10 backdrop-blur-sm border border-background/20">
              <div className="p-3 rounded-full bg-background/20">
                <TrendingDown className="h-6 w-6 text-primary-foreground" />
              </div>
              <div className="text-center">
                <h3 className="font-semibold text-primary-foreground mb-1">Fuel Optimization</h3>
                <p className="text-sm text-primary-foreground/80">Reduce costs by up to 30%</p>
              </div>
            </div>

            <div className="flex flex-col items-center gap-3 p-6 rounded-2xl bg-background/10 backdrop-blur-sm border border-background/20">
              <div className="p-3 rounded-full bg-background/20">
                <Map className="h-6 w-6 text-primary-foreground" />
              </div>
              <div className="text-center">
                <h3 className="font-semibold text-primary-foreground mb-1">Smart Routing</h3>
                <p className="text-sm text-primary-foreground/80">A* & D* Lite algorithms</p>
              </div>
            </div>

            <div className="flex flex-col items-center gap-3 p-6 rounded-2xl bg-background/10 backdrop-blur-sm border border-background/20">
              <div className="p-3 rounded-full bg-background/20">
                <Zap className="h-6 w-6 text-primary-foreground" />
              </div>
              <div className="text-center">
                <h3 className="font-semibold text-primary-foreground mb-1">Real-Time Data</h3>
                <p className="text-sm text-primary-foreground/80">Weather & wave forecasts</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Wave decoration at bottom */}
      <div className="absolute bottom-0 left-0 right-0 h-24 z-10 overflow-hidden">
        <svg viewBox="0 0 1440 120" className="w-full h-full" preserveAspectRatio="none">
          <path
            d="M0,64L48,69.3C96,75,192,85,288,80C384,75,480,53,576,48C672,43,768,53,864,58.7C960,64,1056,64,1152,58.7C1248,53,1344,43,1392,37.3L1440,32L1440,120L1392,120C1344,120,1248,120,1152,120C1056,120,960,120,864,120C768,120,672,120,576,120C480,120,384,120,288,120C192,120,96,120,48,120L0,120Z"
            fill="hsl(var(--background))"
          />
        </svg>
      </div>
    </section>
  );
};
