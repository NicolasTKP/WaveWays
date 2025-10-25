import { Users, Award, Target, Lightbulb } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import oceanPattern from "@/assets/ocean-pattern.jpg";

export const About = () => {
  return (
    <section id="about" className="py-24 bg-background relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-5">
        <img src={oceanPattern} alt="" className="w-full h-full object-cover" />
      </div>

      <div className="container mx-auto px-4 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left Column - Text Content */}
          <div className="space-y-6 animate-slide-in-left">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10">
              <Users className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium text-primary">
                About WaveWays
              </span>
            </div>

            <h2 className="text-4xl md:text-5xl font-bold leading-tight">
              Revolutionizing Maritime{" "}
              <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                Route Planning
              </span>
            </h2>

            <p className="text-lg text-muted-foreground leading-relaxed">
              WaveWays is an advanced vessel route optimization system developed by InternshipSeekers for maritime route planning excellence. Our system combines cutting-edge algorithms with real-time environmental data to deliver the most efficient and cost-effective shipping routes.
            </p>

            <div className="space-y-4">
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold">
                  ✓
                </div>
                <div>
                  <h3 className="font-semibold mb-1">Advanced Pathfinding Algorithms</h3>
                  <p className="text-sm text-muted-foreground">
                    Utilizing A* and D* Lite for dynamic route optimization considering real-time conditions
                  </p>
                </div>
              </div>

              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold">
                  ✓
                </div>
                <div>
                  <h3 className="font-semibold mb-1">Environmental Integration</h3>
                  <p className="text-sm text-muted-foreground">
                    Marine forecasts, wave data, wind conditions, and bathymetry for comprehensive route planning
                  </p>
                </div>
              </div>

              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold">
                  ✓
                </div>
                <div>
                  <h3 className="font-semibold mb-1">Fuel Optimization</h3>
                  <p className="text-sm text-muted-foreground">
                    Minimize operational costs with intelligent fuel consumption calculations and safety margins
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Price List */}
          <div className="w-full max-w-4xl mx-auto bg-white rounded-lg shadow-lg overflow-hidden animate-fade-in">
            <div className="grid grid-cols-3 text-center text-white font-bold text-lg">
              <div className="p-4 bg-blue-600">Allowance Price</div>
              <div className="p-4 bg-blue-700">Normal Price</div>
              <div className="p-4 bg-blue-800">Co-Op Bundle</div>
            </div>
            {/* Rows for pricing details */}
            {/* Row 1 */}
            <div className="grid grid-cols-3 text-center py-4 border-b border-gray-200">
              <div className="p-4 text-blue-800">RM0.05 / minute</div>
              <div className="p-4 text-blue-800">RM0.10 / minute</div>
              <div className="p-4 text-blue-800">RM 27 / 5 hours (10% Discount)</div>
            </div>
            {/* Row 2 */}
            <div className="grid grid-cols-3 text-center py-4 border-b border-gray-200">
              <div className="p-4 text-blue-800">Free RM30 Credits Upon Registration (10 hours)</div>
              <div className="p-4 text-blue-800">Free RM60 Credits Upon Registration (10 hours)</div>
              <div className="p-4 text-blue-800">RM51 / 10 hours (15% Discount)</div>
            </div>
            {/* Row 3 */}
            <div className="grid grid-cols-3 text-center py-4 border-b border-gray-200">
              <div className="p-4 text-blue-800">5% Off Above 80 hours per Month</div>
              <div className="p-4 text-blue-800">5% Off Above 100 hours per Month</div>
              <div className="p-4 text-blue-800">RM 120 / 25 hours (20% Discount)</div>
            </div>
            {/* Row 4 */}
            <div className="grid grid-cols-3 text-center py-4">
              <div className="p-4 text-blue-800">Cap at RM 400 per Month</div>
              <div className="p-4 text-blue-800"></div> {/* Empty cell for Normal Price */}
              <div className="p-4 text-blue-800">RM 225 / 50 hours (25% Discount)</div>
            </div>
          </div>
        </div>

        {/* Technology Stack */}
        <div className="mt-16 pt-16 border-t border-border">
          <div className="text-center mb-12">
            <h3 className="text-2xl font-bold mb-3">Built With Cutting-Edge Technology</h3>
            <p className="text-muted-foreground">
              Powered by advanced algorithms and comprehensive maritime data
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              { name: "A* Algorithm", desc: "Optimal pathfinding" },
              { name: "D* Lite", desc: "Dynamic re-planning" },
              { name: "TSP Solver", desc: "Multi-port optimization" },
              { name: "Marine Data", desc: "Real-time forecasts" },
            ].map((tech, index) => (
              <div
                key={index}
                className="text-center p-6 rounded-xl border-2 border-border hover:border-primary/50 transition-all hover:shadow-lg animate-fade-in"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="font-semibold mb-1">{tech.name}</div>
                <div className="text-sm text-muted-foreground">{tech.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
