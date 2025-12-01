"use client";

import { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

mapboxgl.accessToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || "";
const BLOB_BASE_URL = process.env.NEXT_PUBLIC_BLOB_URL || '';

// Types
interface VolcanoProperties {
  id: string;
  name: string;
  country: string;
  status: string;
  status_simple: string;
  last_eruption_year: number | null;
  risk_score: number;
  risk_category: string;
  predicted_fatalities: number;
  confidence_level: number;
  total_population: number;
  high_risk_population: number;
  economic_loss_usd: number;
  slope_mean: number;
  high_hazard_area_pct: number;
  forest_at_risk_km2: number;
  assessment_date: string;
  assessment_year: number;
  buffer_radius_km: number;
  has_dem_raster: boolean;
  has_lulc_raster: boolean;
}

interface VolcanoFeature {
  type: "Feature";
  geometry: {
    type: "Point";
    coordinates: [number, number];
  };
  properties: VolcanoProperties;
}

interface VolcanoGeoJSON {
  type: "FeatureCollection";
  metadata: {
    title: string;
    generated_at: string;
    total_volcanoes: number;
    successful_assessments: number;
  };
  features: VolcanoFeature[];
}

// Utility functions
const formatNumber = (num: number) => {
  if (num >= 1e9) return `${(num / 1e9).toFixed(1)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
  if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
  return num.toFixed(0);
};

const formatCurrency = (num: number) => {
  if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
  if (num >= 1e3) return `$${(num / 1e3).toFixed(2)}K`;
  return `$${num.toFixed(0)}`;
};

const getRiskColor = (category: string) => {
  switch (category) {
    case "EXTREME": return "#dc2626";
    case "HIGH": return "#ea580c";
    case "MODERATE": return "#f59e0b";
    case "LOW": return "#16a34a";
    default: return "#6b7280";
  }
};

const getRiskGradient = (category: string) => {
  switch (category) {
    case "EXTREME": return "from-red-600 via-red-500 to-orange-500";
    case "HIGH": return "from-orange-600 via-orange-500 to-amber-500";
    case "MODERATE": return "from-amber-500 via-yellow-500 to-yellow-400";
    case "LOW": return "from-green-600 via-green-500 to-emerald-500";
    default: return "from-gray-600 via-gray-500 to-gray-400";
  }
};

export default function VolcanoDashboard() {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<mapboxgl.Map | null>(null);

  const [data, setData] = useState<VolcanoGeoJSON | null>(null);
  const [selected, setSelected] = useState<VolcanoFeature | null>(null);
  const [loading, setLoading] = useState(true);
  const [showDEM, setShowDEM] = useState(false);
  const [showLULC, setShowLULC] = useState(false);
  const [showRivers, setShowRivers] = useState(true);
  const [mapStyle, setMapStyle] = useState<"satellite" | "terrain" | "streets">("satellite");

  // Load volcano data
  useEffect(() => {
    fetch("/data/volcano_risk.json")
      .then((res) => res.json())
      .then((json: VolcanoGeoJSON) => {
        setData(json);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error loading volcano_risk.json", err);
        setLoading(false);
      });
  }, []);

  // Initialize map with advanced features
  useEffect(() => {
    if (!mapContainerRef.current || !data || mapRef.current) return;

    const getMapStyle = () => {
      switch (mapStyle) {
        case "satellite": return "mapbox://styles/mapbox/satellite-streets-v12";
        case "terrain": return "mapbox://styles/mapbox/outdoors-v12";
        case "streets": return "mapbox://styles/mapbox/streets-v12";
        default: return "mapbox://styles/mapbox/satellite-streets-v12";
      }
    };

    const map = new mapboxgl.Map({
      container: mapContainerRef.current,
      style: getMapStyle(),
      center: [110.45, -7.54],
      zoom: 3,
      pitch: 45,
      bearing: 0,
      antialias: true,
      maxPitch: 85,
      minPitch: 0,
    });

    mapRef.current = map;

    const nav = new mapboxgl.NavigationControl({
      visualizePitch: true,
      showCompass: true,
      showZoom: true
    });
    map.addControl(nav, "top-right");

    map.addControl(new mapboxgl.ScaleControl({
      maxWidth: 100,
      unit: "metric"
    }), "bottom-right");

    map.on("load", () => {
      map.addSource("mapbox-dem", {
        type: "raster-dem",
        url: "mapbox://mapbox.mapbox-terrain-dem-v1",
        tileSize: 512,
        maxzoom: 14,
      });
      map.setTerrain({ source: "mapbox-dem", exaggeration: 1.5 });

      map.addLayer({
        id: "sky",
        type: "sky",
        paint: {
          "sky-type": "atmosphere",
          "sky-atmosphere-sun": [0.0, 90.0],
          "sky-atmosphere-sun-intensity": 15,
        },
      });

      if (map.getLayer("water")) {
        map.setPaintProperty("water", "fill-color", "#1e40af");
        map.setPaintProperty("water", "fill-opacity", 0.6);
      }
      
      if (map.getLayer("waterway")) {
        map.setPaintProperty("waterway", "line-color", "#1e40af");
        map.setPaintProperty("waterway", "line-width", 2);
      }

      map.addSource("volcanoes", {
        type: "geojson",
        data: data as any,
      });

      map.addLayer({
        id: "volcano-glow",
        type: "circle",
        source: "volcanoes",
        paint: {
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["zoom"],
            2, 20,
            10, 60,
          ],
          "circle-color": [
            "match",
            ["get", "risk_category"],
            "EXTREME", "#dc2626",
            "HIGH", "#ea580c",
            "MODERATE", "#f59e0b",
            "LOW", "#16a34a",
            "#6b7280",
          ],
          "circle-opacity": [
            "interpolate",
            ["linear"],
            ["zoom"],
            2, 0.15,
            10, 0.15,
            12, 0.05,
            14, 0
          ],
          "circle-blur": 1,
        },
      });

      map.addSource("hydrosheds-rivers", {
        type: "vector",
        url: "mapbox://mapbox.mapbox-streets-v8"
      });

      map.addLayer({
        id: "rivers-flow-layer",
        type: "line",
        source: "hydrosheds-rivers",
        "source-layer": "waterway",
        filter: ["in", "class", "river", "canal"],
        paint: {
          "line-color": [
            "interpolate",
            ["linear"],
            ["zoom"],
            8, "#0ea5e9",
            12, "#0284c7",
            16, "#0369a1"
          ],
          "line-width": [
            "interpolate",
            ["exponential", 1.5],
            ["zoom"],
            8, 2,
            12, 4,
            16, 8
          ],
          "line-opacity": 0.8,
        },
        layout: {
          "line-cap": "round",
          "line-join": "round",
          "visibility": showRivers ? "visible" : "none"
        }
      }, "volcano-glow");

      map.addLayer({
        id: "streams-layer",
        type: "line",
        source: "hydrosheds-rivers",
        "source-layer": "waterway",
        filter: ["in", "class", "stream", "stream_intermittent"],
        paint: {
          "line-color": "#06b6d4",
          "line-width": [
            "interpolate",
            ["linear"],
            ["zoom"],
            10, 0.5,
            14, 2,
            18, 4
          ],
          "line-opacity": 0.6,
        },
        layout: {
          "visibility": showRivers ? "visible" : "none"
        }
      }, "volcano-glow");

      map.addLayer({
        id: "volcano-points",
        type: "circle",
        source: "volcanoes",
        paint: {
          "circle-radius": [
            "interpolate",
            ["exponential", 1.5],
            ["zoom"],
            2, ["*", ["get", "risk_score"], 0.8],
            10, ["*", ["get", "risk_score"], 3],
            12, ["*", ["get", "risk_score"], 2],
            14, 0
          ],
          "circle-color": [
            "match",
            ["get", "risk_category"],
            "EXTREME", "#dc2626",
            "HIGH", "#ea580c",
            "MODERATE", "#f59e0b",
            "LOW", "#16a34a",
            "#6b7280",
          ],
          "circle-stroke-width": [
            "interpolate",
            ["linear"],
            ["zoom"],
            2, 3,
            10, 3,
            12, 2,
            14, 0
          ],
          "circle-stroke-color": "#ffffff",
          "circle-opacity": [
            "interpolate",
            ["linear"],
            ["zoom"],
            2, 0.95,
            10, 0.95,
            11, 0.7,
            12, 0.4,
            13, 0.1,
            14, 0
          ],
        },
      });

      map.addLayer({
        id: "volcano-heatmap",
        type: "heatmap",
        source: "volcanoes",
        maxzoom: 9,
        paint: {
          "heatmap-weight": [
            "interpolate",
            ["linear"],
            ["get", "risk_score"],
            0, 0,
            100, 1,
          ],
          "heatmap-intensity": [
            "interpolate",
            ["linear"],
            ["zoom"],
            0, 1,
            9, 3,
          ],
          "heatmap-color": [
            "interpolate",
            ["linear"],
            ["heatmap-density"],
            0, "rgba(0, 0, 255, 0)",
            0.2, "rgb(103, 169, 207)",
            0.4, "rgb(209, 229, 240)",
            0.6, "rgb(253, 219, 199)",
            0.8, "rgb(239, 138, 98)",
            1, "rgb(178, 24, 43)",
          ],
          "heatmap-radius": [
            "interpolate",
            ["linear"],
            ["zoom"],
            0, 20,
            9, 40,
          ],
          "heatmap-opacity": 0.3,
        },
      }, "volcano-glow");

      map.on("click", "volcano-points", async (e: any) => {
        if (!e.features || e.features.length === 0) return;

        const feature = e.features[0];
        const volcano: VolcanoFeature = {
          type: "Feature",
          geometry: feature.geometry,
          properties: feature.properties,
        };

        setSelected(volcano);

        const [lng, lat] = volcano.geometry.coordinates;
        
        map.flyTo({ 
          center: [lng, lat], 
          zoom: 12, 
          pitch: 60,
          bearing: 30,
          duration: 2000,
          essential: true,
        });

        const radiusKm = 10;
        const points: [number, number][] = [];
        const steps = 128;
        for (let i = 0; i <= steps; i++) {
          const angle = (i / steps) * 2 * Math.PI;
          const dx = (radiusKm / 111.32) * Math.cos(angle);
          const dy = (radiusKm / 110.57) * Math.sin(angle);
          points.push([lng + dx, lat + dy]);
        }

        const bufferGeoJSON = {
          type: "FeatureCollection",
          features: [{
            type: "Feature",
            geometry: {
              type: "Polygon",
              coordinates: [points],
            },
          }],
        };

        if (map.getSource("buffer")) {
          (map.getSource("buffer") as mapboxgl.GeoJSONSource).setData(bufferGeoJSON as any);
        } else {
          map.addSource("buffer", {
            type: "geojson",
            data: bufferGeoJSON as any,
          });
          
          map.addLayer({
            id: "buffer-fill",
            type: "fill",
            source: "buffer",
            paint: {
              "fill-color": getRiskColor(volcano.properties.risk_category),
              "fill-opacity": 0.2,
            },
          }, "volcano-glow");
          
          map.addLayer({
            id: "buffer-outline",
            type: "line",
            source: "buffer",
            paint: {
              "line-color": getRiskColor(volcano.properties.risk_category),
              "line-width": 3,
              "line-dasharray": [2, 4],
              "line-opacity": 0.8,
            },
          });
        }

        await loadRiversForVolcano(map, volcano.properties.name);

        const popup = new mapboxgl.Popup({
          closeButton: false,
          closeOnClick: false,
          offset: 25,
          className: "volcano-popup",
        })
          .setLngLat([lng, lat])
          .setHTML(`
            <div style="
              font-weight: bold; 
              font-size: 14px;
              color: #1f2937;
              text-shadow: 0 1px 2px rgba(255,255,255,0.9), 0 0 8px rgba(255,255,255,0.8);
              padding: 4px 8px;
              background: rgba(255,255,255,0.95);
              border-radius: 6px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            ">
              ${volcano.properties.name}
            </div>
          `)
          .addTo(map);

        await loadRasters(map, volcano.properties.name, lat, lng, radiusKm);
      });

      map.on("mouseenter", "volcano-points", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "volcano-points", () => {
        map.getCanvas().style.cursor = "";
      });
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [data, mapStyle]);

  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;
    
    if (map.getLayer("rivers-flow-layer")) {
      map.setLayoutProperty("rivers-flow-layer", "visibility", showRivers ? "visible" : "none");
    }
    
    if (map.getLayer("streams-layer")) {
      map.setLayoutProperty("streams-layer", "visibility", showRivers ? "visible" : "none");
    }
    
    if (map.getLayer("water")) {
      map.setLayoutProperty("water", "visibility", showRivers ? "visible" : "none");
    }

    if (map.getLayer("hydrosheds-rivers-glow")) {
      map.setLayoutProperty("hydrosheds-rivers-glow", "visibility", showRivers ? "visible" : "none");
    }
    
    if (map.getLayer("hydrosheds-rivers-detailed")) {
      map.setLayoutProperty("hydrosheds-rivers-detailed", "visibility", showRivers ? "visible" : "none");
    }
    
    if (map.getLayer("jrc-water-bodies")) {
      map.setLayoutProperty("jrc-water-bodies", "visibility", showRivers ? "visible" : "none");
    }
  }, [showRivers]);

  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;
    if (map.getLayer("dem-image-layer")) {
      map.setLayoutProperty("dem-image-layer", "visibility", showDEM ? "visible" : "none");
    }
  }, [showDEM]);

  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;
    if (map.getLayer("lulc-image-layer")) {
      map.setLayoutProperty("lulc-image-layer", "visibility", showLULC ? "visible" : "none");
    }
  }, [showLULC]);

  const loadRiversForVolcano = async (
    map: mapboxgl.Map,
    volcanoName: string
  ) => {
    const safeName = volcanoName.replace(/ /g, "_").replace(/\//g, "_");
    const riverUrl = `${BLOB_BASE_URL}/rivers/${safeName}_rivers.geojson`;
    const waterUrl = `${BLOB_BASE_URL}/rivers/${safeName}_water.geojson`;

    const layersToRemove = [
      "hydrosheds-rivers-glow",
      "hydrosheds-rivers-detailed", 
      "jrc-water-bodies"
    ];
    
    layersToRemove.forEach(layerId => {
      if (map.getLayer(layerId)) {
        map.removeLayer(layerId);
      }
    });

    const sourcesToRemove = [
      "hydrosheds-rivers-data",
      "jrc-water-data"
    ];
    
    sourcesToRemove.forEach(sourceId => {
      if (map.getSource(sourceId)) {
        map.removeSource(sourceId);
      }
    });

    try {
      const riverRes = await fetch(riverUrl);
      
      if (riverRes.ok) {
        const riverData = await riverRes.json();

        if (!riverData.type || !riverData.features) {
          throw new Error('Invalid GeoJSON');
        }

        map.addSource("hydrosheds-rivers-data", {
          type: "geojson",
          data: riverData,
        });

        map.addLayer({
          id: "hydrosheds-rivers-glow",
          type: "line",
          source: "hydrosheds-rivers-data",
          paint: {
            "line-color": "#0ea5e9",
            "line-width": [
              "interpolate",
              ["exponential", 1.5],
              ["zoom"],
              8, 8,
              12, 16,
              16, 32
            ],
            "line-opacity": 0.3,
            "line-blur": 4,
          },
          layout: {
            "line-cap": "round",
            "line-join": "round",
            "visibility": showRivers ? "visible" : "none"
          }
        }, "volcano-points");

        map.addLayer({
          id: "hydrosheds-rivers-detailed",
          type: "line",
          source: "hydrosheds-rivers-data",
          paint: {
            "line-color": [
              "interpolate",
              ["linear"],
              ["zoom"],
              8, "#06b6d4",
              12, "#0891b2",
              16, "#0e7490"
            ],
            "line-width": [
              "interpolate",
              ["exponential", 1.5],
              ["zoom"],
              8, 2,
              12, 4,
              16, 8
            ],
            "line-opacity": 0.95,
          },
          layout: {
            "line-cap": "round",
            "line-join": "round",
            "visibility": showRivers ? "visible" : "none"
          }
        }, "volcano-points");

        console.log('✅ Rivers loaded from Blob');
        return;
      }
    } catch (e: any) {
      console.log(`ℹ️ No HydroSHEDS rivers found for ${volcanoName}`);
    }

    try {
      const waterRes = await fetch(waterUrl);
      
      if (waterRes.ok) {
        const waterData = await waterRes.json();

        map.addSource("jrc-water-data", {
          type: "geojson",
          data: waterData,
        });

        map.addLayer({
          id: "jrc-water-bodies",
          type: "fill",
          source: "jrc-water-data",
          paint: {
            "fill-color": "#0ea5e9",
            "fill-opacity": 0.5,
            "fill-outline-color": "#0284c7",
          },
          layout: {
            "visibility": showRivers ? "visible" : "none"
          }
        }, "volcano-points");

        console.log('✅ Water bodies loaded from Blob');
        return;
      }
    } catch (e: any) {
      console.log(`ℹ️ No water bodies found for ${volcanoName}`);
    }
  };

  const loadRasters = async (
  map: mapboxgl.Map,
  volcanoName: string,
  lat: number,
  lng: number,
  radiusKm: number
) => {
  console.log(`🔍 Loading rasters for: ${volcanoName}`);
  
  try {
    // Load the raster mapping
    const mappingRes = await fetch('/data/raster-mapping.json');
    const mapping = await mappingRes.json();
    
    const volcanoMapping = mapping[volcanoName];
    
    if (!volcanoMapping) {
      console.log(`⚠️ No raster mapping found for: ${volcanoName}`);
      return;
    }
    
    const demUrl = volcanoMapping.DEM;
    const lulcUrl = volcanoMapping.LULC;
    
    console.log(`📦 DEM URL: ${demUrl}`);
    console.log(`📦 LULC URL: ${lulcUrl}`);

    const delta = radiusKm / 111.0;
    
    const coordinates: [[number, number], [number, number], [number, number], [number, number]] = [
      [lng - delta, lat + delta],
      [lng + delta, lat + delta],
      [lng + delta, lat - delta],
      [lng - delta, lat - delta],
    ];

    // Remove existing layers/sources
    if (map.getLayer("dem-image-layer")) map.removeLayer("dem-image-layer");
    if (map.getSource("dem-image")) map.removeSource("dem-image");
    if (map.getLayer("lulc-image-layer")) map.removeLayer("lulc-image-layer");
    if (map.getSource("lulc-image")) map.removeSource("lulc-image");

    // Load DEM if available
    if (demUrl) {
      console.log(`📦 Loading DEM layer from: ${demUrl}`);
      
      try {
        map.addSource("dem-image", {
          type: "image",
          url: demUrl,
          coordinates: coordinates,
        });
        map.addLayer({
          id: "dem-image-layer",
          type: "raster",
          source: "dem-image",
          paint: { "raster-opacity": 0.7 },
          layout: { visibility: showDEM ? "visible" : "none" },
        }, "volcano-points");
        console.log(`✅ DEM layer added`);
      } catch (error) {
        console.log(`❌ Error loading DEM:`, error);
      }
    }

    // Load LULC if available
    if (lulcUrl) {
      console.log(`📦 Loading LULC layer from: ${lulcUrl}`);
      
      try {
        map.addSource("lulc-image", {
          type: "image",
          url: lulcUrl,
          coordinates: coordinates,
        });
        map.addLayer({
          id: "lulc-image-layer",
          type: "raster",
          source: "lulc-image",
          paint: { "raster-opacity": 0.6 },
          layout: { visibility: showLULC ? "visible" : "none" },
        }, "volcano-points");
        console.log(`✅ LULC layer added`);
      } catch (error) {
        console.log(`❌ Error loading LULC:`, error);
      }
    }
    
    console.log('✅ Raster loading completed');
    
  } catch (error) {
    console.log('❌ Error in loadRasters:', error);
  }
};

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="text-center">
          <div className="text-6xl mb-4 animate-bounce">🌋</div>
          <div className="text-white text-2xl font-bold mb-2">Loading Volcano Data</div>
          <div className="text-gray-400">Analyzing global risk assessments...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-screen bg-slate-900 overflow-hidden">
      {/* Map Container with Controls Overlay */}
      <div className="flex-1 relative">
        <div ref={mapContainerRef} className="w-full h-full" />
        
        {/* Map Style Switcher - Floating Top Left */}
        <div className="absolute top-4 left-4 bg-white rounded-xl shadow-2xl p-2 flex gap-2 z-10">
          <button
            onClick={() => setMapStyle("satellite")}
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all ${
              mapStyle === "satellite" 
                ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg" 
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            🛰️ Satellite
          </button>
          <button
            onClick={() => setMapStyle("terrain")}
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all ${
              mapStyle === "terrain" 
                ? "bg-gradient-to-r from-green-600 to-green-500 text-white shadow-lg" 
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            ⛰️ Terrain
          </button>
          <button
            onClick={() => setMapStyle("streets")}
            className={`px-4 py-2 rounded-lg font-semibold text-sm transition-all ${
              mapStyle === "streets" 
                ? "bg-gradient-to-r from-purple-600 to-purple-500 text-white shadow-lg" 
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            🗺️ Streets
          </button>
        </div>

        {/* Legend - Floating Bottom Left */}
        <div className="absolute bottom-8 left-4 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 rounded-2xl shadow-2xl p-4 z-10 max-w-xs border-2 border-slate-700">
          <h4 className="font-bold text-xs text-white mb-2 uppercase tracking-wide flex items-center gap-2">
            <span className="text-base">🎯</span> Risk Levels
          </h4>
          <div className="space-y-1.5">
            {[
              { level: "EXTREME", color: "#dc2626", icon: "🔴" },
              { level: "HIGH", color: "#ea580c", icon: "🟠" },
              { level: "MODERATE", color: "#f59e0b", icon: "🟡" },
              { level: "LOW", color: "#16a34a", icon: "🟢" },
            ].map(({ level, color, icon }) => (
              <div key={level} className="flex items-center gap-2 group">
                <div className="text-sm">{icon}</div>
                <div 
                  className="w-4 h-4 rounded shadow-lg group-hover:scale-110 transition-transform" 
                  style={{ backgroundColor: color }}
                />
                <div className="font-bold text-xs text-white">{level}</div>
              </div>
            ))}
          </div>
          
          <div className="mt-3 pt-3 border-t border-slate-700">
            <h4 className="font-bold text-xs text-white mb-2 uppercase tracking-wide flex items-center gap-2">
              <span className="text-base">💧</span> Water
            </h4>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <div className="w-8 h-2 bg-gradient-to-r from-cyan-400 to-blue-600 rounded-full shadow-md"></div>
                <span className="text-xs text-slate-300">Rivers</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-8 h-1.5 bg-cyan-400 rounded-full shadow-md"></div>
                <span className="text-xs text-slate-400">Streams</span>
              </div>
            </div>
          </div>

          {/* LULC Legend - Updated with colored boxes */}
          <div className="mt-3 pt-3 border-t border-slate-700">
            <h4 className="font-bold text-xs text-white mb-2 uppercase tracking-wide flex items-center gap-2">
              <span className="text-base">🗺️</span> LULC Classes
            </h4>
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { name: "Forest", color: "#0d5f0d" },
                { name: "Grassland", color: "#b5d96a" },
                { name: "Cropland", color: "#ffdb5c" },
                { name: "Urban", color: "#cc0013" },
                { name: "Water", color: "#0000ff" },
                { name: "Barren", color: "#969696" },
              ].map(({ name, color }) => (
                <div key={name} className="flex items-center gap-1.5">
                  <div 
                    className="w-3 h-3 rounded-sm shadow-md border border-white/30" 
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-xs text-slate-300">{name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* 3D Navigation Guide */}
          <div className="mt-3 pt-3 border-t border-slate-700">
            <h4 className="font-bold text-xs text-white mb-2 uppercase tracking-wide flex items-center gap-2">
              <span className="text-base">🎮</span> 3D Controls
            </h4>
            <div className="space-y-1 text-xs text-slate-300">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-cyan-400 min-w-[70px]">Right-click:</span>
                <span>Rotate 360°</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-cyan-400 min-w-[70px]">Ctrl + Drag:</span>
                <span>Tilt view</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-cyan-400 min-w-[70px]">Scroll:</span>
                <span>Zoom</span>
              </div>
            </div>
          </div>
        </div>

        {/* Interactive Cursor Hint - Bottom Center */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 z-10">
          <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 rounded-full shadow-2xl px-6 py-3 flex items-center gap-3 animate-pulse border-2 border-white/30">
            <span className="text-2xl">🖱️</span>
            <div className="text-white font-bold text-sm">
              <span className="block text-xs opacity-90">Try it now!</span>
              Right-click + Drag for 3D View
            </div>
          </div>
        </div>
      </div>

      {/* Professional Sidebar */}
      <div className="w-[480px] bg-white shadow-2xl flex flex-col overflow-hidden">
        {/* Gradient Header */}
        <div className="bg-gradient-to-br from-red-600 via-orange-600 to-amber-600 p-6 text-white shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="text-5xl">🌋</div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">Volcano Risk Monitor</h1>
                <p className="text-white/90 text-sm font-medium">Global Assessment System</p>
              </div>
            </div>
          </div>
          {data && (
            <div className="flex items-center gap-4 text-sm">
              <div className="bg-white/20 backdrop-blur-sm rounded-lg px-4 py-2 font-semibold">
                📊 {data.metadata.total_volcanoes} Volcanoes
              </div>
              <div className="bg-white/20 backdrop-blur-sm rounded-lg px-4 py-2 font-semibold">
                🌍 Global Coverage
              </div>
            </div>
          )}
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto">
          {selected ? (
            <div className="p-6 space-y-6">
              {/* Volcano Title - Updated Format */}
              <div className="border-b-2 border-gray-100 pb-5">
                <h2 className="text-3xl font-bold text-gray-900 mb-2 leading-tight">
                  {selected.properties.name}
                </h2>
                <div className="flex flex-col gap-1">
                  {/* Country - Bold */}
                  <div className="flex items-center gap-2">
                    <span className="text-lg">📍</span>
                    <span className="font-bold text-gray-800">{selected.properties.country}</span>
                  </div>
                  
                  {/* Status - Bold */}
                  {selected.properties.status_simple && (
                    <div className="flex items-center gap-2">
                      <span className="text-lg">🌋</span>
                      <span className="font-bold text-gray-800">Status: {selected.properties.status_simple}</span>
                    </div>
                  )}
                  
                  {/* Last Eruption - Bold */}
                  {selected.properties.last_eruption_year && selected.properties.last_eruption_year > 0 ? (
                    <div className="flex items-center gap-2">
                      <span className="text-lg">🔥</span>
                      <span className="font-bold text-red-600">Last Eruption: {selected.properties.last_eruption_year}</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <span className="text-lg">🔥</span>
                      <span className="font-bold text-gray-600">Last Eruption: Unknown</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Risk Score - Hero Card */}
              <div 
                className={`relative overflow-hidden rounded-2xl p-8 text-white shadow-2xl bg-gradient-to-br ${getRiskGradient(selected.properties.risk_category)}`}
              >
                <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16"></div>
                <div className="absolute bottom-0 left-0 w-24 h-24 bg-black/10 rounded-full -ml-12 -mb-12"></div>
                
                <div className="relative z-10">
                  <div className="text-xs font-bold mb-3 opacity-90 uppercase tracking-widest">Risk Assessment</div>
                  <div className="flex items-end gap-4 mb-4">
                    <div className="text-7xl font-black">
                      {selected.properties.risk_score.toFixed(1)}
                    </div>
                    <div className="text-3xl font-bold mb-2">
                      / 100
                    </div>
                  </div>
                  <div className="text-2xl font-bold tracking-wide">
                    {selected.properties.risk_category} RISK
                  </div>
                </div>
              </div>

              {/* Impact Grid */}
              <div className="grid grid-cols-2 gap-4">
                {/* Population Card */}
                <div className="col-span-2 bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-5 border-2 border-purple-200">
                  <div className="flex items-center gap-2 mb-4">
                    <span className="text-2xl">👥</span>
                    <h3 className="font-bold text-gray-900 uppercase text-sm tracking-wide">Population Impact</h3>
                  </div>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600 text-sm font-medium">Total Population</span>
                      <span className="text-2xl font-bold text-gray-900">
                        {formatNumber(selected.properties.total_population)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600 text-sm font-medium">High Risk Zone</span>
                      <span className="text-2xl font-bold text-orange-600">
                        {formatNumber(selected.properties.high_risk_population)}
                      </span>
                    </div>
                    <div className="pt-3 border-t-2 border-purple-200">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-700 text-sm font-bold">Predicted Fatalities</span>
                        <span className="text-3xl font-black text-red-600">
                          {formatNumber(selected.properties.predicted_fatalities)}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Economic Loss */}
                <div className="bg-gradient-to-br from-green-50 to-emerald-100 rounded-xl p-5 border-2 border-green-200">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-xl">💰</span>
                    <h3 className="font-bold text-gray-900 uppercase text-xs tracking-wide">Economic</h3>
                  </div>
                  <div className="text-xs text-gray-600 mb-2 font-medium">Estimated Loss</div>
                  <div className="text-2xl font-black text-gray-900">
                    {formatCurrency(selected.properties.economic_loss_usd)}
                  </div>
                </div>

                {/* Forest Impact */}
                <div className="bg-gradient-to-br from-amber-50 to-yellow-100 rounded-xl p-5 border-2 border-amber-200">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-xl">🌳</span>
                    <h3 className="font-bold text-gray-900 uppercase text-xs tracking-wide">Forest</h3>
                  </div>
                  <div className="text-xs text-gray-600 mb-2 font-medium">Area at Risk</div>
                  <div className="text-2xl font-black text-gray-900">
                    {selected.properties.forest_at_risk_km2.toFixed(1)} km²
                  </div>
                </div>
              </div>

              {/* Terrain Analysis */}
              <div className="bg-gradient-to-br from-blue-50 to-cyan-100 rounded-xl p-5 border-2 border-blue-200">
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-2xl">⛰️</span>
                  <h3 className="font-bold text-gray-900 uppercase text-sm tracking-wide">Terrain Hazards</h3>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-gray-600 mb-1 font-medium">Mean Slope</div>
                    <div className="text-3xl font-bold text-gray-900">
                      {selected.properties.slope_mean.toFixed(1)}°
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-600 mb-1 font-medium">High Hazard Area</div>
                    <div className="text-3xl font-bold text-orange-600">
                      {selected.properties.high_hazard_area_pct.toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>

              {/* Data Layers */}
              <div className="bg-gray-50 rounded-xl p-5 border-2 border-gray-200">
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-2xl">🗺️</span>
                  <h3 className="font-bold text-gray-900 uppercase text-sm tracking-wide">Data Layers</h3>
                </div>
                <div className="space-y-3">
                  <label className="flex items-center justify-between p-3 bg-white rounded-lg cursor-pointer hover:shadow-md transition-all group border border-gray-200">
                    <div className="flex items-center gap-3">
                      <div className="w-6 h-6 bg-gradient-to-r from-blue-400 via-cyan-500 to-blue-600 rounded-md shadow-md"></div>
                      <div>
                        <div className="text-sm font-bold text-gray-900 group-hover:text-cyan-600">Rivers & Waterways</div>
                        <div className="text-xs text-gray-500">HydroSHEDS + JRC Water</div>
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      checked={showRivers}
                      onChange={(e) => setShowRivers(e.target.checked)}
                      className="w-6 h-6 accent-cyan-600 cursor-pointer"
                    />
                  </label>

                  <label className="flex items-center justify-between p-3 bg-white rounded-lg cursor-pointer hover:shadow-md transition-all group border border-gray-200">
                    <div className="flex items-center gap-3">
                      <div className="w-6 h-6 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-md shadow-md"></div>
                      <div>
                        <div className="text-sm font-bold text-gray-900 group-hover:text-blue-600">DEM Elevation</div>
                        <div className="text-xs text-gray-500">Digital Elevation Model</div>
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      checked={showDEM}
                      onChange={(e) => setShowDEM(e.target.checked)}
                      className="w-6 h-6 accent-blue-600 cursor-pointer"
                    />
                  </label>
                  
                  <label className="flex items-center justify-between p-3 bg-white rounded-lg cursor-pointer hover:shadow-md transition-all group border border-gray-200">
                    <div className="flex items-center gap-3">
                      <div className="w-6 h-6 bg-gradient-to-r from-blue-500 via-green-500 to-yellow-500 rounded-md shadow-md"></div>
                      <div>
                        <div className="text-sm font-bold text-gray-900 group-hover:text-green-600">LULC Land Cover</div>
                        <div className="text-xs text-gray-500">Land Use Classification</div>
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      checked={showLULC}
                      onChange={(e) => setShowLULC(e.target.checked)}
                      className="w-6 h-6 accent-green-600 cursor-pointer"
                    />
                  </label>
                </div>
              </div>

              {/* Assessment Metadata */}
              <div className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl p-5 border-2 border-slate-200">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xl">📋</span>
                  <h3 className="font-bold text-gray-900 uppercase text-xs tracking-wide">Assessment Info</h3>
                </div>
                <div className="space-y-2 text-xs text-gray-600">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-gray-900">📅 Date:</span>
                    <span>{new Date(selected.properties.assessment_date).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-gray-900">📏 Buffer:</span>
                    <span>{selected.properties.buffer_radius_km} km radius</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-gray-900">🤖 Method:</span>
                    <span>TerraMind AI + Multi-Criteria Analysis</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-6">
              {/* Empty State */}
              <div className="bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50 rounded-2xl p-12 text-center border-2 border-gray-200">
                <div className="text-7xl mb-6 animate-bounce">🌋</div>
                <h3 className="text-2xl font-bold text-gray-900 mb-3">Select a Volcano</h3>
                <p className="text-gray-600 leading-relaxed max-w-sm mx-auto">
                  Click on any volcano marker on the map to view comprehensive risk assessment, impact analysis, and terrain data
                </p>
              </div>

              {/* Global Statistics */}
              {data && (
                <div className="mt-8 space-y-6">
                  <div className="flex items-center gap-2 mb-4">
                    <span className="text-2xl">🌍</span>
                    <h3 className="text-xl font-bold text-gray-900 uppercase tracking-wide">Global Overview</h3>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    {["EXTREME", "HIGH", "MODERATE", "LOW"].map(category => {
                      const count = data.features.filter(
                        f => f.properties.risk_category === category
                      ).length;
                      const percentage = ((count / data.metadata.total_volcanoes) * 100).toFixed(1);
                      
                      return (
                        <div
                          key={category}
                          className="bg-white rounded-xl p-5 border-l-4 shadow-lg hover:shadow-xl transition-shadow"
                          style={{ borderLeftColor: getRiskColor(category) }}
                        >
                          <div className="text-xs text-gray-500 mb-2 font-bold uppercase tracking-wider">{category}</div>
                          <div className="text-4xl font-black text-gray-900 mb-1">{count}</div>
                          <div className="text-xs text-gray-600 font-semibold">{percentage}% of total</div>
                          <div className="mt-3 w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="h-2 rounded-full transition-all duration-500"
                              style={{ 
                                width: `${percentage}%`,
                                backgroundColor: getRiskColor(category)
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Total Statistics */}
                  <div className="bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 rounded-xl p-6 text-white shadow-xl">
                    <div className="flex items-center justify-between mb-4">
                      <div className="text-sm font-bold uppercase tracking-wider opacity-90">Total Volcanoes</div>
                      <div className="text-4xl font-black">{data.metadata.total_volcanoes}</div>
                    </div>
                    <div className="text-xs opacity-90 leading-relaxed">
                      Comprehensive risk analysis across {data.metadata.total_volcanoes} active and dormant volcanoes worldwide using advanced AI and multi-criteria decision analysis
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
