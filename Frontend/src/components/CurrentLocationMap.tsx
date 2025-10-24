import React, { useEffect, useState, useRef } from "react";
import { GoogleMap, LoadScript, Marker } from "@react-google-maps/api";
import { Compass } from "lucide-react";
import { Port } from "@/types/route";

interface CurrentLocationMapProps {
  currentLocation: { lat: number; lng: number };
  ports?: Port[];
  onMapClick?: (lat: number, lng: number) => void;
}

const containerStyle = {
  width: "100%",
  height: "400px",
  borderRadius: "12px",
  overflow: "hidden",
};

const CurrentLocationMap: React.FC<CurrentLocationMapProps> = ({ currentLocation }) => {
  const [userLocation, setUserLocation] = useState<google.maps.LatLngLiteral | null>(currentLocation || null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const mapRef = useRef<google.maps.Map | null>(null);

  // 🧭 Auto-track user location (real-time updates)
  useEffect(() => {
    if (!navigator.geolocation) {
      console.warn("Geolocation not supported by this browser.");
      return;
    }

    const watchId = navigator.geolocation.watchPosition(
      (position) => {
        const newLocation = {
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        };
        setUserLocation(newLocation);
        if (mapRef.current) mapRef.current.panTo(newLocation);
      },
      (error) => console.error("Error tracking location:", error),
      { enableHighAccuracy: true, maximumAge: 10000, timeout: 20000 }
    );

    return () => navigator.geolocation.clearWatch(watchId);
  }, []);

  const onLoad = (map: google.maps.Map) => {
    mapRef.current = map;
    setMapLoaded(true);
    if (userLocation) {
      map.panTo(userLocation);
      map.setZoom(14);
    }
  };

  const handleRecenter = () => {
    if (mapRef.current && userLocation) {
      mapRef.current.panTo(userLocation);
      mapRef.current.setZoom(14);
    }
  };

  return (
    <div className="relative bg-white rounded-lg shadow-md border border-gray-200 p-4">
      <h2 className="text-lg font-semibold text-[#023e8a] mb-2">
        Current Vessel Location
      </h2>

      {/* 🧭 Load Google Map */}
      <LoadScript
        googleMapsApiKey={import.meta.env.VITE_GOOGLE_MAPS_API_KEY!}
        onLoad={() => setMapLoaded(true)}
      >
        {!mapLoaded ? (
          <div className="flex items-center justify-center h-[400px] text-gray-500 text-sm">
            🛰️ Initializing Google Map...
          </div>
        ) : (
          <GoogleMap
            mapContainerStyle={containerStyle}
            onLoad={onLoad}
            options={{
              streetViewControl: false,
              mapTypeControl: false,
              fullscreenControl: false,
              zoomControl: true,
              gestureHandling: "greedy",
            }}
          >
            {userLocation && (
              <Marker
                position={userLocation}
                title={`Your Vessel (${userLocation.lat.toFixed(4)}, ${userLocation.lng.toFixed(4)})`}
                icon={{
                  url: "https://cdn-icons-png.flaticon.com/512/3531/3531769.png",
                  scaledSize: new google.maps.Size(40, 40),
                }}
              />
            )}
          </GoogleMap>
        )}
      </LoadScript>

      {/* 📍 Coordinates */}
      <div className="mt-3 text-sm text-gray-600">
        {userLocation ? (
          <>
            Latitude: <span className="font-semibold">{userLocation.lat.toFixed(6)}</span> &nbsp;|&nbsp;
            Longitude: <span className="font-semibold">{userLocation.lng.toFixed(6)}</span>
          </>
        ) : (
          "Locating your position..."
        )}
      </div>

      {/* 🧭 Recenter */}
      <button
        onClick={handleRecenter}
        className="absolute bottom-6 right-6 bg-[#0077b6] text-white px-4 py-2 rounded-full shadow-lg hover:bg-[#00b4d8] transition flex items-center gap-2"
      >
        <Compass className="h-4 w-4" />
        Recenter
      </button>
    </div>
  );
};

export default CurrentLocationMap;
