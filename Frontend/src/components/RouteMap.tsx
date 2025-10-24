import { Navigation } from 'lucide-react';
import { GoogleMap, Marker, Polyline, useJsApiLoader } from "@react-google-maps/api";
import { useEffect, useState } from "react";


interface Port {
  name: string;
  lat: number;
  lng: number;
  country?: string;
}

interface RouteSegment {
  start: Port;
  end: Port;
  path: [number, number][];
  distance: number;
  fuelConsumption: number;
}

interface RouteMapProps {
  ports?: Port[];
  route?: RouteSegment[];
  currentLocation?: Port;
  center?: google.maps.LatLngLiteral;
  zoom?: number;
  onMapClick?: (lat: number, lng: number) => void;
}

export const RouteMap = ({
  ports = [],
  route = [],
  center = { lat: 4.2105, lng: 101.9758 }, // Malaysia center
  zoom =4,
  onMapClick // <-- new callback
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
      absolute top-14 left-12 z-50
      flex items-center gap-2
      px-4 py-2
      bg-gradient-to-r from-primary to-accent
      text-white font-semibold
      rounded-lg
      shadow-lg
      hover:from-primary/90 hover:to-accent/90
      transition-all
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
        {/* Current vessel marker */}
        {currentLocation && (
          <Marker
            key="current-location"
            position={{ lat: currentLocation.lat, lng: currentLocation.lng }}
            title={`You Are Here`}  // optional tooltip
          />
        )}

        {/* Ports */}
        {ports.map((port, i) => (
          <Marker
            key={port.name + i} // ← changed to avoid duplicate keys
            position={{ lat: port.lat, lng: port.lng }}
            title={port.name}
          />
        ))}

        {/* Route lines */}
        {route.map((segment, index) => (
          <Polyline
            key={index}
            path={segment.path.map(([lat, lng]) => ({ lat, lng }))}
            options={{
              strokeColor: index === 0 ? "#0ea5e9" : "#22c55e",
              strokeOpacity: 0.9,
              strokeWeight: 2.5,
              geodesic: true,
            }}
          />
        ))}
      </GoogleMap>
    </div>
  );
};
