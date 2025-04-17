"use client"

import React, { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Sidebar, SidebarSection, SidebarItem } from "@/components/ui/sidebar"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { Activity, AlertTriangle, Navigation, Search, Home, Shield, Globe, Cpu, BarChart2, FileText, Cog, Zap, Bot, Wind, Compass, Anchor, Gauge, Radio, Database, GitBranch, Box, Waves, Clock, Brain, Sliders } from 'lucide-react'

// Main dashboard component for maritime vessel tracking and AI control system
export default function Dashboard() {
  // State for real-time sensor data from vessel (simulated)
  const [sensorData, setSensorData] = useState({
    gps: { lat: 0, lon: 0 }, // GPS coordinates
    wind: { speed: 0, direction: 0 }, // Wind speed and direction
    current: { speed: 0, direction: 0 }, // Water current speed and direction
    gyroscope: { pitch: 0, roll: 0, yaw: 0 }, // Vessel orientation
    engineMetrics: { rpm: 0, temperature: 0, fuelConsumption: 0 }, // Engine performance
    rudderPosition: 0 // Current rudder angle
  })

  // State for ML model feature vectors (simulated)
  const [features, setFeatures] = useState({
    position: [], // Position-related features (e.g., scaled lat/lon, distance from port)
    motion: [], // Motion and dynamics features (e.g., speed, acceleration, turn rate)
    environmental: [], // Environmental condition features (e.g., wind, current impact)
    historical: [] // Historical trajectory features (e.g., past positions, common routes)
  })

  // State for AI model performance metrics (simulated)
  const [modelPerformance, setModelPerformance] = useState({
    accuracy: 0, // Overall prediction accuracy
    confidence: 0 // Model confidence score for current prediction
  })

  // State for AI steering recommendations (simulated)
  const [steeringRecommendation, setSteeringRecommendation] = useState(0) // Recommended rudder angle

  // State for historical performance data of different ML models (simulated)
  const [data, setData] = useState([
    // Each entry contains timestamp and accuracy/performance scores for different models
    { timestamp: 1678886400000, logistic: 0.8, kalman: 0.7, rl: 0.6, lstm: 0.9 },
    // ... additional historical data points would be added here
  ]);

  // State for detected anomalies in vessel operations (simulated)
  const [anomalies, setAnomalies] = useState([]); // Stores recent anomaly events

  // Effect hook to initialize and update the performance chart data periodically
  useEffect(() => {
    // Generate initial set of data points for the chart
    const initialData = Array.from({ length: 20 }, (_, i) => generateDataPoint(Date.now() - (20 - i) * 1000)); // Start with past data
    setData(initialData);

    // Set up an interval to add new data points and remove old ones
    const interval = setInterval(() => {
      const newPoint = generateDataPoint(Date.now()); // Generate a new data point for the current time
      setData(prevData => [...prevData.slice(1), newPoint]); // Add new point, remove oldest
      detectAnomalies(newPoint); // Check the new data point for potential anomalies
    }, 1000); // Update every second

    // Cleanup function to clear the interval when the component unmounts
    return () => clearInterval(interval);
  }, []); // Empty dependency array ensures this runs only once on mount

  // Generates a simulated data point with realistic vessel metrics and model performance
  const generateDataPoint = (timestamp) => ({
    timestamp: timestamp, // Timestamp for the data point
    // Simulated model performance scores with sinusoidal variation + noise
    logistic: Math.sin(timestamp / 10000) * 10 + 70 + Math.random() * 2, // Example: Logistic Regression performance
    kalman: Math.sin(timestamp / 10000) * 10 + 75 + Math.random(),     // Example: Kalman Filter performance
    rl: Math.sin(timestamp / 10000) * 10 + 80 + Math.random() * 1.5,   // Example: Reinforcement Learning performance
    lstm: Math.sin(timestamp / 10000) * 10 + 85 + Math.random() * 0.8,    // Example: LSTM performance

    // Simulated vessel metrics with sinusoidal variation
    speed: 15 + Math.sin(timestamp / 8000) * 3, // Vessel speed in knots
    heading: 180 + Math.sin(timestamp / 15000) * 10, // Vessel heading in degrees

    // Simulated engine telemetry
    engineLoad: 75 + Math.sin(timestamp / 12000) * 5, // Engine load percentage
    temperature: 85 + Math.sin(timestamp / 9000) * 3, // Engine temperature in Celsius

    // Simulated environmental conditions
    windSpeed: 12 + Math.sin(timestamp / 20000) * 4, // Wind speed in knots
    waveHeight: 2 + Math.sin(timestamp / 18000) // Wave height in meters
  });

  // Detects anomalies based on simple thresholds for engine metrics
  const detectAnomalies = (point) => {
    // Check if engine load or temperature exceeds predefined thresholds
    if (point.engineLoad > 85 || point.temperature > 90) {
      // Add the anomaly to the list, keeping only the last 10 anomalies
      setAnomalies(prev => [...prev.slice(-10), {
        timestamp: point.timestamp, // Timestamp of the anomaly
        value: Math.max(point.engineLoad, point.temperature), // Value that triggered the anomaly
        type: point.engineLoad > 85 ? 'High Load' : 'High Temp' // Type of anomaly
      }]);
    }
  };

  // Effect hook to update simulated sensor data, features, performance, and steering periodically
  useEffect(() => {
    // Set up an interval to update various dashboard states
    const interval = setInterval(() => {
      updateSensorData(); // Update simulated sensor readings
      updateFeatures(); // Update simulated feature vectors
      updateModelPerformance(); // Update simulated model performance metrics
      updateSteeringRecommendation(); // Update simulated steering recommendation
    }, 1000); // Update every second

    // Cleanup function to clear the interval when the component unmounts
    return () => clearInterval(interval);
  }, []); // Empty dependency array ensures this runs only once on mount

  // Updates the sensor data state with new random values (simulation)
  const updateSensorData = () => {
    setSensorData({
      gps: { lat: Math.random() * 180 - 90, lon: Math.random() * 360 - 180 },
      wind: { speed: Math.random() * 50, direction: Math.random() * 360 },
      current: { speed: Math.random() * 10, direction: Math.random() * 360 },
      gyroscope: { pitch: Math.random() * 10 - 5, roll: Math.random() * 10 - 5, yaw: Math.random() * 360 },
      engineMetrics: { rpm: Math.random() * 3000, temperature: Math.random() * 100 + 50, fuelConsumption: Math.random() * 50 },
      rudderPosition: Math.random() * 70 - 35 // Angle from -35 to +35 degrees
    });
  }

  // Updates the feature vector state with new random values (simulation)
  const updateFeatures = () => {
    setFeatures({
      position: [Math.random(), Math.random(), Math.random()], // Example: 3 position features
      motion: [Math.random(), Math.random(), Math.random()], // Example: 3 motion features
      environmental: [Math.random(), Math.random(), Math.random()], // Example: 3 environmental features
      historical: [Math.random(), Math.random(), Math.random()] // Example: 3 historical features
    });
  }

  // Updates the model performance state with new random values (simulation)
  const updateModelPerformance = () => {
    setModelPerformance({
      accuracy: 0.85 + Math.random() * 0.1, // Accuracy between 0.85 and 0.95
      confidence: 0.8 + Math.random() * 0.15 // Confidence between 0.8 and 0.95
    });
  }

  // Updates the steering recommendation state with a new random value (simulation)
  const updateSteeringRecommendation = () => {
    setSteeringRecommendation(Math.random() * 70 - 35); // Recommend angle between -35 and +35 degrees
  }

  // Render the main dashboard layout
  return (
    <div className="flex h-screen bg-[#0f0a19] text-white">
      <Sidebar>
        <SidebarSection className="p-4">
          <div className="flex items-center space-x-2 mb-6">
            <div className="w-8 h-8 bg-blue-500 rounded-md flex items-center justify-center text-xl font-bold">
              M
            </div>
            <div>
              <div className="font-semibold">Maritime AI</div>
              <div className="text-xs text-gray-400">captain@maritime-ai.com</div>
            </div>
          </div>
          <div className="relative">
            <Input
              type="text"
              placeholder="Search"
              className="w-full bg-[#1c1528] border-none pl-8"
            />
            <Search className="absolute left-2 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          </div>
        </SidebarSection>
        <SidebarSection>
          <SidebarItem href="#" active><Home className="mr-2 h-4 w-4" /> Dashboard</SidebarItem>
        </SidebarSection>
        <SidebarSection>
          <div className="text-xs font-semibold text-gray-500 mb-2">SENSORS</div>
          <SidebarItem href="#"><Navigation className="mr-2 h-4 w-4" /> GPS</SidebarItem>
          <SidebarItem href="#"><Wind className="mr-2 h-4 w-4" /> Wind Sensors</SidebarItem>
          <SidebarItem href="#"><Waves className="mr-2 h-4 w-4" /> Current Sensors</SidebarItem>
          <SidebarItem href="#"><Compass className="mr-2 h-4 w-4" /> Gyroscope</SidebarItem>
          <SidebarItem href="#"><Gauge className="mr-2 h-4 w-4" /> Engine Metrics</SidebarItem>
          <SidebarItem href="#"><Anchor className="mr-2 h-4 w-4" /> Rudder Position</SidebarItem>
        </SidebarSection>
        <SidebarSection>
          <div className="text-xs font-semibold text-gray-500 mb-2">SYSTEM</div>
          <SidebarItem href="#"><Database className="mr-2 h-4 w-4" /> Raw Data</SidebarItem>
          <SidebarItem href="#"><GitBranch className="mr-2 h-4 w-4" /> Features</SidebarItem>
          <SidebarItem href="#"><Brain className="mr-2 h-4 w-4" /> Model Training</SidebarItem>
          <SidebarItem href="#"><Zap className="mr-2 h-4 w-4" /> Real-time Prediction</SidebarItem>
          <SidebarItem href="#"><Sliders className="mr-2 h-4 w-4" /> Steering Control</SidebarItem>
        </SidebarSection>
      </Sidebar>
      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-[1200px] mx-auto space-y-6">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-3xl font-bold">Maritime AI Control System</h1>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <SensorCard title="GPS" icon={<Navigation className="w-6 h-6" />} data={sensorData.gps} />
            <SensorCard title="Wind" icon={<Wind className="w-6 h-6" />} data={sensorData.wind} />
            <SensorCard title="Current" icon={<Waves className="w-6 h-6" />} data={sensorData.current} />
            <SensorCard title="Gyroscope" icon={<Compass className="w-6 h-6" />} data={sensorData.gyroscope} />
            <SensorCard title="Engine Metrics" icon={<Gauge className="w-6 h-6" />} data={sensorData.engineMetrics} />
            <SensorCard title="Rudder Position" icon={<Anchor className="w-6 h-6" />} data={{ position: sensorData.rudderPosition }} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="bg-[#1c1528] border-none">
              <CardHeader>
                <CardTitle className="text-xl">Feature Engineering</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-4">
                  <FeatureCard title="Position" data={features.position} />
                  <FeatureCard title="Motion" data={features.motion} />
                  <FeatureCard title="Environmental" data={features.environmental} />
                  <FeatureCard title="Historical" data={features.historical} />
                </div>
              </CardContent>
            </Card>

           {/* Model Performance */}
        <Card className="bg-gray-900 border-blue-500 border">
          <CardHeader>
            <CardTitle className="text-xl text-blue-400">Model Performance Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <div style={{ width: '100%', height: 300 }}>
              <ResponsiveContainer>
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2d374d" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={ts => new Date(ts).toLocaleTimeString()}
                    stroke="#fff" 
                  />
                  <YAxis stroke="#fff" />
                  <Tooltip 
                    labelFormatter={ts => new Date(ts).toLocaleTimeString()}
                    contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid #666' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="logistic" stroke="#ff0080" name="Logistic" dot={false} />
                  <Line type="monotone" dataKey="kalman" stroke="#00ff80" name="Kalman" dot={false} />
                  <Line type="monotone" dataKey="rl" stroke="#0080ff" name="RL" dot={false} />
                  <Line type="monotone" dataKey="lstm" stroke="#ff8000" name="LSTM" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
          </div>

          <Card className="bg-[#1c1528] border-none">
            <CardHeader>
              <CardTitle className="text-xl">Steering Control</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-center">
                <div className="relative w-64 h-64">
                  <svg className="w-full h-full" viewBox="0 0 100 100">
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      fill="none"
                      stroke="#3f3356"
                      strokeWidth="10"
                    />
                    <line
                      x1="50"
                      y1="50"
                      x2="50"
                      y2="10"
                      stroke="#fbbf24"
                      strokeWidth="4"
                      transform={`rotate(${steeringRecommendation} 50 50)`}
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-3xl font-bold">{steeringRecommendation.toFixed(1)}°</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}

// Helper components for displaying sensor data and features
function SensorCard({ title, icon, data }) {
  return (
    <Card className="bg-[#1c1528]/80 border border-[#3f3356]">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-[#f8f9fa]">{title}</h3>
          <div className="bg-[#2d2a3d]/60 p-2 rounded-full">
            {React.cloneElement(icon, { className: "w-6 h-6 text-[#61dafb]" })}
          </div>
        </div>
        <div className="space-y-2">
          {Object.entries(data).map(([key, value]) => (
            <div key={key} className="flex justify-between">
              <span className="text-[#d1d5db]">{key}</span>
              <span className="text-[#f8f9fa]">{typeof value === 'number' ? value.toFixed(2) : value}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function FeatureCard({ title, data }) {
  return (
    <div className="bg-[#2d2a3d]/60 p-4 rounded-lg border border-[#3f3356]">
      <h4 className="text-sm font-semibold mb-2 text-[#f8f9fa]">{title}</h4>
      <div className="space-y-1">
        {data.map((value, index) => (
          <div key={index} className="w-full bg-[#1c1528]/80 rounded-full h-2">
            <div
              className="bg-[#61dafb] h-2 rounded-full"
              style={{ width: `${value * 100}%` }}
            ></div>
          </div>
        ))}
      </div>
    </div>
  )
}
