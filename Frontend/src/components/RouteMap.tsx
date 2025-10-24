import { Navigation } from 'lucide-react';
import { GoogleMap, Marker, Polyline, Polygon, useJsApiLoader } from "@react-google-maps/api";
import { useEffect, useState } from "react";
import { PointModel } from "@/services/routeService"; // Import PointModel

interface Port {
  name: string;
  lat: number;
  lng: number;
  country?: string;
}

interface RouteSegmentWithColor {
  id: string; // Unique ID for the route
  start: Port;
  end: Port;
  path: [number, number][];
  distance?: number; // Optional for D* Lite path
  fuelConsumption?: number; // Optional for D* Lite path
  color: string; // Color for the polyline
}

interface RouteMapProps {
  ports?: Port[];
  routes?: RouteSegmentWithColor[]; // Changed to routes (plural)
  currentLocation?: Port;
  center?: google.maps.LatLngLiteral;
  zoom?: number;
  onMapClick?: (lat: number, lng: number) => void;
  obstaclePolygon?: PointModel[]; // New prop for obstacle
}

export const RouteMap = ({
  ports = [],
  routes = [], // Changed to routes (plural)
  center = { lat: 4.2105, lng: 101.9758 }, // Malaysia center
  zoom = 4,
  onMapClick,
  obstaclePolygon = [], // Default to empty array
}: RouteMapProps) => {
  const { isLoaded } = useJsApiLoader({
    googleMapsApiKey: import.meta.env.VITE_GOOGLE_MAPS_API_KEY!,
  });

  const [map, setMap] = useState<google.maps.Map | null>(null);
  const [currentLocation, setCurrentLocation] = useState<Port | null>(null);

  // Get browser location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const { latitude, longitude } = pos.coords;
          setCurrentLocation({ name: "You Are Here", lat: latitude, lng: longitude });
        },
        (err) => {
          console.warn("Geolocation failed, using default center", err);
          setCurrentLocation({ name: "Default Location", lat: center.lat, lng: center.lng });
        }
      );
    } else {
      setCurrentLocation({ name: "Default Location", lat: center.lat, lng: center.lng });
    }
  }, [center.lat, center.lng]);

  useEffect(() => {
    if (map && currentLocation) {
      // Recenter the map to current location
      map.panTo({ lat: currentLocation.lat, lng: currentLocation.lng });
      map.setZoom(6); // adjust zoom for Malaysia view
    }
  }, [map, currentLocation]);


  if (!isLoaded) return <div>Loading map...</div>;

  return (
    <div className="w-full h-[500px] rounded-lg overflow-hidden border">
      <button
        onClick={() => {
          if (map && currentLocation) {
            map.panTo({ lat: currentLocation.lat, lng: currentLocation.lng });
            map.setZoom(6);
          }
        }}
        className="
      absolute z-50
      flex items-center gap-2
      px-4 py-2
      bg-gradient-to-r from-primary to-accent
      text-white font-semibold
      rounded-lg
      shadow-lg
      hover:from-primary/90 hover:to-accent/90
      transition-all transform translate-x-[1dvw] translate-y-[7dvh]
    "
      >
        <span className="inline-block">🚢 Recenter</span>
      </button>
      <GoogleMap
        center={currentLocation ? { lat: currentLocation.lat, lng: currentLocation.lng } : center}
        zoom={zoom}
        mapContainerStyle={{ width: "100%", height: "100%" }}
        onLoad={(map) => setMap(map)}
        onClick={(e) => {
          if (!e.latLng || !onMapClick) return;
          const lat = e.latLng.lat();
          const lng = e.latLng.lng();
          onMapClick(lat, lng);
        }}
      >

        {/* Ports */}
        {ports.map((port, i) => (
          <Marker
            key={port.name + i} // ← changed to avoid duplicate keys
            position={{ lat: port.lat, lng: port.lng }}
            title={port.name}
          />
        ))}

        {/* Route lines */}
        {routes.map((routeSegment) => (
          <Polyline
            key={routeSegment.id}
            path={routeSegment.path.map(([lat, lng]) => ({ lat, lng }))}
            options={{
              strokeColor: routeSegment.color,
              strokeOpacity: 0.9,
              strokeWeight: 2.5,
              geodesic: true,
            }}
          />
        ))}

        {/* Obstacle Polygon */}
        {obstaclePolygon.length > 0 && (
          <Polygon
            paths={obstaclePolygon.map((p) => ({ lat: p.lat, lng: p.lon }))}
            options={{
              strokeColor: "#FF0000",
              strokeOpacity: 0.8,
              strokeWeight: 2,
              fillColor: "#FF0000",
              fillOpacity: 0.35,
            }}
          />
        )}
      </GoogleMap>
    </div>
  );
};
