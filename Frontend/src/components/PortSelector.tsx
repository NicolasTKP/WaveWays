import { useState, useEffect } from 'react';
import { Search, MapPin, X, Plus } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card } from '@/components/ui/card';
import { Port } from '@/types/route';
import { routeApi } from '@/services/routeApi';

interface PortSelectorProps {
  selectedPorts: Port[];
  onPortsChange: (ports: Port[]) => void;
}

export const PortSelector = ({ selectedPorts, onPortsChange }: PortSelectorProps) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [availablePorts, setAvailablePorts] = useState<Port[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadPorts();
  }, []);

  const loadPorts = async () => {
    setIsLoading(true);
    const ports = await routeApi.getPorts();
    setAvailablePorts(ports);
    setIsLoading(false);
  };

  const handleSearch = async (query: string) => {
    setSearchQuery(query);
    if (query.length > 2) {
      const results = await routeApi.searchPorts(query);
      setAvailablePorts(results);
    } else {
      loadPorts();
    }
  };

  const addPort = (port: Port) => {
    if (!selectedPorts.find(p => p.id === port.id)) {
      onPortsChange([...selectedPorts, port]);
    }
  };

  const removePort = (portId: string) => {
    onPortsChange(selectedPorts.filter(p => p.id !== portId));
  };

  const filteredPorts = availablePorts.filter(
    port => !selectedPorts.find(p => p.id === port.id)
  );

  return (
    <div className="space-y-4">
      {/* Selected Ports */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Selected Ports ({selectedPorts.length})</label>
          {selectedPorts.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onPortsChange([])}
              className="h-8 text-xs"
            >
              Clear All
            </Button>
          )}
        </div>
        
        {selectedPorts.length === 0 ? (
          <Card className="p-4 border-dashed">
            <p className="text-sm text-muted-foreground text-center">
              No ports selected. Search and add ports below.
            </p>
          </Card>
        ) : (
          <div className="space-y-2">
            {selectedPorts.map((port, index) => (
              <div
                key={port.id}
                className="flex items-center justify-between p-3 rounded-lg border bg-card/50 hover:bg-card transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Badge
                    variant={index === 0 ? 'default' : index === selectedPorts.length - 1 ? 'destructive' : 'secondary'}
                    className="min-w-[60px] justify-center"
                  >
                    {index === 0 ? 'Start' : index === selectedPorts.length - 1 ? 'End' : `Stop ${index}`}
                  </Badge>
                  <div>
                    <p className="font-medium text-sm">{port.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {port.country} • {port.lat.toFixed(4)}°, {port.lng.toFixed(4)}°
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removePort(port.id)}
                  className="h-8 w-8 p-0"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Port Search */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Add Ports</label>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search ports by name..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            className="pl-10"
          />
        </div>
        
        <ScrollArea className="h-[200px] rounded-lg border bg-card/30">
          <div className="p-2 space-y-1">
            {isLoading ? (
              <p className="text-sm text-muted-foreground text-center py-4">Loading ports...</p>
            ) : filteredPorts.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">
                {searchQuery ? 'No ports found' : 'All ports selected'}
              </p>
            ) : (
              filteredPorts.map(port => (
                <button
                  key={port.id}
                  onClick={() => addPort(port)}
                  className="w-full flex items-center justify-between p-2 rounded-md hover:bg-accent/10 transition-colors text-left group"
                >
                  <div className="flex items-center gap-2">
                    <MapPin className="h-4 w-4 text-accent flex-shrink-0" />
                    <div>
                      <p className="text-sm font-medium">{port.name}</p>
                      <p className="text-xs text-muted-foreground">{port.country}</p>
                    </div>
                  </div>
                  <Plus className="h-4 w-4 text-muted-foreground group-hover:text-accent transition-colors" />
                </button>
              ))
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
};
