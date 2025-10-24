import React, { useEffect, useState, useRef } from "react";
import { GoogleMap, LoadScript, Marker } from "@react-google-maps/api";

interface MapVisualProps { //to accept data
  ports?: any[];
  route?: any[];
  currentLocation?: any;
}


const containerStyle = {
  width: "100%",
  height: "400px",
  borderRadius: "12px",
  overflow: "hidden",
};

// Malaysia region boundaries (East & West)
const malaysiaBounds = {
  north: 7.5,
  south: 0.8,
  west: 99.5,
  east: 119.5,
};

const MapVisual: React.FC<MapVisualProps> = ({ ports = [], route = [], currentLocation }) => {
  const [userLocation, setUserLocation] = useState<google.maps.LatLngLiteral | null>(null);
  const mapRef = useRef<google.maps.Map | null>(null);

  // 🧭 Get user’s current location
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
        },
        
        (error) => console.warn("Geolocation failed:", error.message)
      );
    } else {
      console.warn("Geolocation not supported by this browser.");
    }
  }, []);

  // 🗺️ When map loads, fit Malaysia region
  const onLoad = (map: google.maps.Map) => {
    mapRef.current = map;
    const bounds = new google.maps.LatLngBounds(
      { lat: malaysiaBounds.south, lng: malaysiaBounds.west },
      { lat: malaysiaBounds.north, lng: malaysiaBounds.east }
    );
    map.fitBounds(bounds);
  };

  // 🎯 Recenter map back to Malaysia
  const handleRecenter = () => {
    if (mapRef.current) {
      const bounds = new google.maps.LatLngBounds(
        { lat: malaysiaBounds.south, lng: malaysiaBounds.west },
        { lat: malaysiaBounds.north, lng: malaysiaBounds.east }
      );
      mapRef.current.fitBounds(bounds);
    }
  };

  return (
    <div className="relative bg-white rounded-lg shadow-md border border-gray-200 p-4">
      <h2 className="text-lg font-semibold text-[#023e8a] mb-2">
        Malaysian Ports Coverage Map
      </h2>
    
      <LoadScript googleMapsApiKey={import.meta.env.VITE_GOOGLE_MAPS_API_KEY!}>
        <GoogleMap
          mapContainerStyle={containerStyle}
          onLoad={onLoad}
          options={{
            streetViewControl: false,
            mapTypeControl: false,
            fullscreenControl: true,
            zoomControl: true,
          }}
        >
          {/* 📍 Current user location */}
          {userLocation && <Marker position={userLocation} title="Your Location" />}


        </GoogleMap>
      </LoadScript>

      {/* Recenter button top-left inside map */}
        <button
          onClick={handleRecenter}
          className="absolute top-4 left-4 z-50 bg-[#0077b6] text-white px-4 py-2 rounded-full shadow-lg hover:bg-[#00b4d8] transition"
        >
        Recenter Map
      </button>
    </div>
  );
};

export default MapVisual;
