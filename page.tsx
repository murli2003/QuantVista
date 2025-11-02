"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  Shield,
  BarChart3,
  Zap,
  AlertTriangle,
  ExternalLink,
  Wifi,
  WifiOff,
  Clock,
  Calculator,
  Calendar,
  CheckCircle,
  XCircle,
} from "lucide-react"

interface MarketStatus {
  is_open: boolean
  status: string
  message: string
  next_open?: string
  next_close?: string
  current_time: string
}

interface TradingCalendar {
  next_trading_days: Array<{
    date: string
    day: string
    formatted: string
  }>
  upcoming_holidays: Array<{
    date: string
    day: string
    formatted: string
  }>
}

interface CandleData {
  current: number
  last_close: number
  today_open: number
  percent: number
  candles: number[][]
  signals?: TradingSignal[]
  final_signal?: FinalSignal
  indicators?: any
  options_chain?: OptionData[]
  portfolio_greeks?: PortfolioGreeks
  market_status?: MarketStatus
  trading_calendar?: TradingCalendar
  data_source?: string
  timestamp?: string
}

interface TradingSignal {
  id: string
  name: string
  signal: "BUY" | "SELL" | "NEUTRAL"
  value: string
  strength: number
  description: string
  category: "technical" | "options" | "momentum" | "structure"
}

interface FinalSignal {
  action: "BUY" | "SELL" | "HOLD"
  confidence: number
  entry: number
  stopLoss: number
  target1: number
  target2: number
  riskReward: number
  positionSize: string
  reasoning: string[]
}

interface OptionData {
  strike: number
  expiry: string
  time_to_expiry: number
  call: {
    symbol: string
    price: number
    iv: number
    delta: number
    gamma: number
    theta: number
    vega: number
    rho: number
    volume: number
    oi: number
    ltp: number
  }
  put: {
    symbol: string
    price: number
    iv: number
    delta: number
    gamma: number
    theta: number
    vega: number
    rho: number
    volume: number
    oi: number
    ltp: number
  }
}

interface PortfolioGreeks {
  total_call_delta: number
  total_put_delta: number
  net_delta: number
  total_gamma: number
  total_theta: number
  total_vega: number
  put_call_ratio: number
  max_pain: number
  total_call_volume: number
  total_put_volume: number
}

const INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY"]

export default function AdvancedTradingDashboard() {
  const [activeIndex, setActiveIndex] = useState("NIFTY")
  const [data, setData] = useState<Record<string, CandleData>>({})
  const [connectionStatus, setConnectionStatus] = useState<Record<string, "connected" | "disconnected" | "error">>({})
  const [lastUpdate, setLastUpdate] = useState<Record<string, string>>({})
  const [showOptionsChain, setShowOptionsChain] = useState(false)
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null)
  const [tradingCalendar, setTradingCalendar] = useState<TradingCalendar | null>(null)
  const wsRefs = useRef<Record<string, WebSocket | null>>({})

  // Initialize connection status
  useEffect(() => {
    const initialConnectionStatus: Record<string, "connected" | "disconnected" | "error"> = {}
    INDICES.forEach((index) => {
      initialConnectionStatus[index] = "disconnected"
    })
    setConnectionStatus(initialConnectionStatus)
  }, [])

  // Update the WebSocket connection useEffect to connect to all indices for quick stats
  useEffect(() => {
    // Connect to all indices for quick stats
    INDICES.forEach((index) => {
      if (!wsRefs.current[index]) {
        connectWebSocket(index)
      }
    })

    return () => {
      // Cleanup on unmount
      Object.values(wsRefs.current).forEach((ws) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.close()
        }
      })
    }
  }, []) // Remove activeIndex dependency

  function connectWebSocket(index: string) {
    try {
      const ws = new WebSocket(`ws://localhost:8000/ws/${index}`)
      wsRefs.current[index] = ws

      ws.onopen = () => {
        console.log(`Connected to ${index} WebSocket`)
        setConnectionStatus((prev) => ({
          ...prev,
          [index]: "connected",
        }))
      }

      ws.onmessage = (event) => {
        try {
          const newData = JSON.parse(event.data)
          if (newData.error) {
            console.error(`Error from ${index} WebSocket:`, newData.error)
            setConnectionStatus((prev) => ({
              ...prev,
              [index]: "error",
            }))
          } else {
            setData((prev) => ({
              ...prev,
              [index]: newData,
            }))
            setLastUpdate((prev) => ({
              ...prev,
              [index]: new Date().toLocaleTimeString(),
            }))

            // Update market status and trading calendar from any source
            if (newData.market_status) {
              setMarketStatus(newData.market_status)
            }
            if (newData.trading_calendar) {
              setTradingCalendar(newData.trading_calendar)
            }
          }
        } catch (err) {
          console.error(`Error parsing WebSocket data for ${index}:`, err)
        }
      }

      ws.onerror = (error) => {
        console.error(`WebSocket error for ${index}:`, error)
        setConnectionStatus((prev) => ({
          ...prev,
          [index]: "error",
        }))
      }

      ws.onclose = () => {
        console.log(`WebSocket closed for ${index}`)
        setConnectionStatus((prev) => ({
          ...prev,
          [index]: "disconnected",
        }))
        wsRefs.current[index] = null

        // Auto-reconnect after 5 seconds
        setTimeout(() => {
          if (!wsRefs.current[index]) {
            connectWebSocket(index)
          }
        }, 5000)
      }
    } catch (err) {
      console.error(`Failed to create WebSocket for ${index}:`, err)
      setConnectionStatus((prev) => ({
        ...prev,
        [index]: "error",
      }))
    }
  }

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case "BUY":
        return "bg-green-500"
      case "SELL":
        return "bg-red-500"
      default:
        return "bg-yellow-500"
    }
  }

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case "BUY":
        return <TrendingUp className="h-4 w-4" />
      case "SELL":
        return <TrendingDown className="h-4 w-4" />
      default:
        return <Minus className="h-4 w-4" />
    }
  }

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case "technical":
        return <BarChart3 className="h-4 w-4" />
      case "options":
        return <Calculator className="h-4 w-4" />
      case "momentum":
        return <Zap className="h-4 w-4" />
      case "structure":
        return <Target className="h-4 w-4" />
      default:
        return <BarChart3 className="h-4 w-4" />
    }
  }

  const getConnectionIcon = (status: string) => {
    switch (status) {
      case "connected":
        return <Wifi className="h-4 w-4 text-green-400" />
      case "error":
        return <AlertTriangle className="h-4 w-4 text-red-400" />
      default:
        return <WifiOff className="h-4 w-4 text-gray-400" />
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "text-green-400"
    if (confidence >= 60) return "text-yellow-400"
    return "text-red-400"
  }

  const formatGreek = (value: number, decimals = 2) => {
    return value.toFixed(decimals)
  }

  const getGreekColor = (value: number, type: string) => {
    if (type === "delta") {
      return value > 0 ? "text-green-400" : "text-red-400"
    }
    if (type === "gamma" || type === "vega") {
      return value > 50 ? "text-yellow-400" : "text-gray-400"
    }
    if (type === "theta") {
      return value < -10 ? "text-red-400" : "text-yellow-400"
    }
    return "text-white"
  }

  const getMarketStatusColor = (status: string) => {
    switch (status) {
      case "Market Open":
        return "text-green-400"
      case "Pre-Market":
        return "text-yellow-400"
      case "Post-Market":
        return "text-orange-400"
      default:
        return "text-red-400"
    }
  }

  const getMarketStatusIcon = (isOpen: boolean) => {
    return isOpen ? <CheckCircle className="h-5 w-5 text-green-400" /> : <XCircle className="h-5 w-5 text-red-400" />
  }

  const isNearATM = (strike: number, currentPrice: number) => {
    const diff = Math.abs(strike - currentPrice)
    return diff <= 100 // Within 100 points of current price
  }

  return (
    <div className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-white mb-2">Advanced Trading Signals with Options Greeks</h1>
          <p className="text-gray-400">
            High-precision signals powered by Angel One API with comprehensive options analysis
          </p>
          <div className="flex items-center gap-2 mt-2 text-sm text-gray-500">
            <Clock className="h-4 w-4" />
            <span>Real-time options Greeks • Delta, Gamma, Theta, Vega analysis</span>
          </div>
        </div>

        {/* Market Status Card */}
        {marketStatus && (
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                {getMarketStatusIcon(marketStatus.is_open)}
                NSE Market Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="bg-gray-700 p-3 rounded">
                  <div className="text-gray-400 text-sm">Status</div>
                  <div className={`text-lg font-bold ${getMarketStatusColor(marketStatus.status)}`}>
                    {marketStatus.status}
                  </div>
                </div>
                <div className="bg-gray-700 p-3 rounded">
                  <div className="text-gray-400 text-sm">Current Time</div>
                  <div className="text-white font-medium">{marketStatus.current_time}</div>
                </div>
                <div className="bg-gray-700 p-3 rounded">
                  <div className="text-gray-400 text-sm">Message</div>
                  <div className="text-white text-sm">{marketStatus.message}</div>
                </div>
                <div className="bg-gray-700 p-3 rounded">
                  <div className="text-gray-400 text-sm">{marketStatus.is_open ? "Closes" : "Next Open"}</div>
                  <div className="text-white text-sm">
                    {marketStatus.is_open ? marketStatus.next_close : marketStatus.next_open}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Trading Calendar */}
        {tradingCalendar && (
          <Card className="bg-gray-800 border-gray-700 mb-6">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Calendar className="h-5 w-5" />
                Trading Calendar
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-white font-semibold mb-3">Next Trading Days</h4>
                  <div className="space-y-2">
                    {tradingCalendar.next_trading_days.slice(0, 5).map((day, idx) => (
                      <div key={idx} className="flex justify-between items-center bg-gray-700 p-2 rounded">
                        <span className="text-white">{day.day}</span>
                        <span className="text-gray-400">{day.formatted}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 className="text-white font-semibold mb-3">Upcoming Holidays</h4>
                  <div className="space-y-2">
                    {tradingCalendar.upcoming_holidays.map((holiday, idx) => (
                      <div key={idx} className="flex justify-between items-center bg-red-900/30 p-2 rounded">
                        <span className="text-red-400">{holiday.day}</span>
                        <span className="text-red-300">{holiday.formatted}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Chart Access Cards */}
        <div className="mb-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {INDICES.map((index) => (
              <Card
                key={index}
                className="bg-gray-800 border-gray-700 hover:bg-gray-700 transition-colors cursor-pointer"
              >
                <CardContent className="p-4">
                  <a
                    href={`/chart/${index}`}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center justify-between text-white hover:text-blue-400"
                  >
                    <div className="flex items-center gap-3">
                      <BarChart3 className="h-5 w-5" />
                      <div>
                        <div className="font-medium">{index} Chart</div>
                        <div className="text-sm text-gray-400">Technical Analysis</div>
                      </div>
                    </div>
                    <ExternalLink className="h-4 w-4" />
                  </a>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        <Tabs value={activeIndex} onValueChange={setActiveIndex} className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-gray-800">
            {INDICES.map((index) => (
              <TabsTrigger key={index} value={index} className="text-white flex items-center gap-2">
                {index}
                {getConnectionIcon(connectionStatus[index])}
                {connectionStatus[index] === "connected" && (
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                )}
              </TabsTrigger>
            ))}
          </TabsList>

          {INDICES.map((index) => (
            <TabsContent key={index} value={index} className="space-y-6">
              {data[index] ? (
                <>
                  {/* Current Price Display */}
                  <Card className="bg-gray-800 border-gray-700">
                    <CardHeader>
                      <CardTitle className="text-white flex items-center justify-between">
                        <span>{index} - Live Price</span>
                        <div className="flex items-center gap-2">
                          <Badge variant={data[index]?.percent >= 0 ? "default" : "destructive"}>
                            {data[index]?.percent ? `${data[index].percent.toFixed(2)}%` : "--"}
                          </Badge>
                          <Badge className="bg-blue-600">
                            <div className="flex items-center gap-1">
                              <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                              LIVE
                            </div>
                          </Badge>
                          {!marketStatus?.is_open && <Badge className="bg-red-600">MARKET CLOSED</Badge>}
                        </div>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold text-white mb-2">
                        ₹{data[index]?.current?.toFixed(2) || "--"}
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        {lastUpdate[index] && <div className="text-gray-400">Last updated: {lastUpdate[index]}</div>}
                        {data[index]?.final_signal && (
                          <div className={`font-bold ${getConfidenceColor(data[index].final_signal!.confidence)}`}>
                            Confidence: {data[index].final_signal!.confidence.toFixed(0)}%
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Options Greeks Summary */}
                  {data[index]?.portfolio_greeks && (
                    <Card className="bg-gray-800 border-gray-700">
                      <CardHeader>
                        <CardTitle className="text-white flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Calculator className="h-5 w-5" />
                            Options Greeks Portfolio Summary
                          </div>
                          <button
                            onClick={() => setShowOptionsChain(!showOptionsChain)}
                            className="text-blue-400 hover:text-blue-300 text-sm"
                          >
                            {showOptionsChain ? "Hide" : "Show"} Full Chain
                          </button>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
                          <div className="bg-gray-700 p-3 rounded">
                            <div className="text-gray-400 text-sm">Net Delta</div>
                            <div
                              className={`text-xl font-bold ${getGreekColor(data[index].portfolio_greeks!.net_delta, "delta")}`}
                            >
                              {formatGreek(data[index].portfolio_greeks!.net_delta, 0)}
                            </div>
                          </div>
                          <div className="bg-gray-700 p-3 rounded">
                            <div className="text-gray-400 text-sm">Total Gamma</div>
                            <div
                              className={`text-xl font-bold ${getGreekColor(data[index].portfolio_greeks!.total_gamma, "gamma")}`}
                            >
                              {formatGreek(data[index].portfolio_greeks!.total_gamma, 1)}
                            </div>
                          </div>
                          <div className="bg-gray-700 p-3 rounded">
                            <div className="text-gray-400 text-sm">Total Theta</div>
                            <div
                              className={`text-xl font-bold ${getGreekColor(data[index].portfolio_greeks!.total_theta, "theta")}`}
                            >
                              {formatGreek(data[index].portfolio_greeks!.total_theta, 1)}
                            </div>
                          </div>
                          <div className="bg-gray-700 p-3 rounded">
                            <div className="text-gray-400 text-sm">Total Vega</div>
                            <div
                              className={`text-xl font-bold ${getGreekColor(data[index].portfolio_greeks!.total_vega, "vega")}`}
                            >
                              {formatGreek(data[index].portfolio_greeks!.total_vega, 1)}
                            </div>
                          </div>
                          <div className="bg-gray-700 p-3 rounded">
                            <div className="text-gray-400 text-sm">Put-Call Ratio</div>
                            <div className="text-xl font-bold text-white">
                              {formatGreek(data[index].portfolio_greeks!.put_call_ratio, 2)}
                            </div>
                          </div>
                          <div className="bg-gray-700 p-3 rounded">
                            <div className="text-gray-400 text-sm">Max Pain</div>
                            <div className="text-xl font-bold text-yellow-400">
                              ₹{formatGreek(data[index].portfolio_greeks!.max_pain, 0)}
                            </div>
                          </div>
                          <div className="bg-gray-700 p-3 rounded">
                            <div className="text-gray-400 text-sm">Call Volume</div>
                            <div className="text-lg font-bold text-green-400">
                              {formatGreek(data[index].portfolio_greeks!.total_call_volume, 0)}
                            </div>
                          </div>
                          <div className="bg-gray-700 p-3 rounded">
                            <div className="text-gray-400 text-sm">Put Volume</div>
                            <div className="text-lg font-bold text-red-400">
                              {formatGreek(data[index].portfolio_greeks!.total_put_volume, 0)}
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* Professional Options Chain Table */}
                  {showOptionsChain && data[index]?.options_chain && data[index].options_chain!.length > 0 && (
                    <Card className="bg-gray-800 border-gray-700">
                      <CardHeader>
                        <CardTitle className="text-white flex items-center gap-2">
                          <Calculator className="h-5 w-5" />
                          Options Chain - {index}
                          <Badge className="bg-blue-600 text-xs">
                            ATM IV:{" "}
                            {data[index].options_chain
                              ?.find((opt) => isNearATM(opt.strike, data[index].current))
                              ?.call.iv.toFixed(1) || "N/A"}
                            %
                          </Badge>
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="overflow-x-auto">
                          <div className="min-w-full">
                            {/* Header */}
                            <div className="grid grid-cols-11 gap-1 text-xs font-semibold text-gray-400 mb-2 px-2">
                              <div className="text-center">Vega</div>
                              <div className="text-center">Gamma</div>
                              <div className="text-center">Theta</div>
                              <div className="text-center">Delta</div>
                              <div className="text-center text-green-400">CALL</div>
                              <div className="text-center text-white font-bold">Strike</div>
                              <div className="text-center text-red-400">PUT</div>
                              <div className="text-center">Delta</div>
                              <div className="text-center">Theta</div>
                              <div className="text-center">Gamma</div>
                              <div className="text-center">Vega</div>
                            </div>

                            {/* Options Data */}
                            <div className="space-y-1">
                              {data[index].options_chain!.map((option, idx) => {
                                const isATM = isNearATM(option.strike, data[index].current)
                                const rowClass = isATM
                                  ? "bg-blue-900/30 border-l-4 border-l-blue-500"
                                  : "hover:bg-gray-700/50"

                                return (
                                  <div
                                    key={idx}
                                    className={`grid grid-cols-11 gap-1 text-xs py-2 px-2 rounded ${rowClass} relative`}
                                  >
                                    {/* Current price indicator line */}
                                    {isATM && (
                                      <div className="absolute left-0 right-0 top-1/2 h-0.5 bg-red-500 opacity-60 z-10"></div>
                                    )}

                                    {/* Call Greeks */}
                                    <div className="text-center text-yellow-400">
                                      {formatGreek(option.call.vega, 1)}
                                    </div>
                                    <div className="text-center text-yellow-400">
                                      {formatGreek(option.call.gamma, 3)}
                                    </div>
                                    <div className="text-center text-red-400">{formatGreek(option.call.theta, 1)}</div>
                                    <div className="text-center text-green-400">
                                      {formatGreek(option.call.delta, 2)}
                                    </div>

                                    {/* Call Price */}
                                    <div className="text-center text-white font-medium">
                                      {option.call.price > 0 ? `₹${formatGreek(option.call.price, 1)}` : "-"}
                                    </div>

                                    {/* Strike Price */}
                                    <div
                                      className={`text-center font-bold ${isATM ? "text-blue-400 bg-blue-900/50 rounded px-1" : "text-white"}`}
                                    >
                                      {option.strike}
                                    </div>

                                    {/* Put Price */}
                                    <div className="text-center text-white font-medium">
                                      {option.put.price > 0 ? `₹${formatGreek(option.put.price, 1)}` : "-"}
                                    </div>

                                    {/* Put Greeks */}
                                    <div className="text-center text-red-400">{formatGreek(option.put.delta, 2)}</div>
                                    <div className="text-center text-red-400">{formatGreek(option.put.theta, 1)}</div>
                                    <div className="text-center text-yellow-400">
                                      {formatGreek(option.put.gamma, 3)}
                                    </div>
                                    <div className="text-center text-yellow-400">{formatGreek(option.put.vega, 1)}</div>
                                  </div>
                                )
                              })}
                            </div>

                            {/* Legend */}
                            <div className="mt-4 text-xs text-gray-400 flex items-center gap-4">
                              <div className="flex items-center gap-1">
                                <div className="w-3 h-0.5 bg-red-500"></div>
                                <span>Current Price: ₹{data[index].current.toFixed(2)}</span>
                              </div>
                              <div className="flex items-center gap-1">
                                <div className="w-3 h-3 bg-blue-900/50 rounded"></div>
                                <span>ATM Strike</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}

                  {/* High-Accuracy Signal Grid */}
                  {data[index]?.signals && data[index].signals!.length > 0 && (
                    <>
                      <div className="text-white text-lg font-semibold mb-4 flex items-center gap-2">
                        <Target className="h-5 w-5" />
                        High-Accuracy Trading Signals ({data[index].signals!.length} indicators)
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                        {data[index].signals!.map((signal) => (
                          <Card
                            key={signal.id}
                            className={`${getSignalColor(signal.signal)} border-none transition-all duration-300 hover:scale-105 hover:shadow-lg`}
                          >
                            <CardHeader className="pb-2">
                              <div className="flex items-center justify-between">
                                <CardTitle className="text-sm font-medium text-white flex items-center gap-2">
                                  {getCategoryIcon(signal.category)}
                                  {signal.name}
                                </CardTitle>
                                <div className="flex items-center gap-1">
                                  {getSignalIcon(signal.signal)}
                                  {signal.strength >= 80 && (
                                    <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                                  )}
                                </div>
                              </div>
                            </CardHeader>
                            <CardContent className="pt-0">
                              <div className="text-xl font-bold text-white mb-2">{signal.value}</div>
                              <Progress value={signal.strength} className="mb-2 h-2" />
                              <div className="text-xs text-white opacity-90 mb-1">{signal.description}</div>
                              <div className="flex items-center justify-between text-xs text-white">
                                <span>Strength: {signal.strength.toFixed(0)}%</span>
                                {signal.strength >= 85 && (
                                  <Badge className="bg-white text-black text-xs px-1 py-0">HIGH</Badge>
                                )}
                              </div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </>
                  )}

                  {/* Final High-Accuracy Signal */}
                  {data[index]?.final_signal && (
                    <Card className={`${getSignalColor(data[index].final_signal!.action)} border-none shadow-xl`}>
                      <CardHeader>
                        <CardTitle className="text-white text-xl flex items-center gap-2">
                          <Target className="h-6 w-6" />
                          Final Trading Signal - {data[index].final_signal!.action}
                          {data[index].final_signal!.confidence >= 80 && (
                            <Badge className="bg-white text-black ml-2">HIGH CONFIDENCE</Badge>
                          )}
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="text-white">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                          <div className="bg-black bg-opacity-20 p-3 rounded">
                            <div className="text-sm opacity-80">Confidence</div>
                            <div
                              className={`text-2xl font-bold ${getConfidenceColor(data[index].final_signal!.confidence)}`}
                            >
                              {data[index].final_signal!.confidence.toFixed(0)}%
                            </div>
                          </div>
                          <div className="bg-black bg-opacity-20 p-3 rounded">
                            <div className="text-sm opacity-80">Entry Price</div>
                            <div className="text-xl font-bold">₹{data[index].final_signal!.entry.toFixed(2)}</div>
                          </div>
                          <div className="bg-black bg-opacity-20 p-3 rounded">
                            <div className="text-sm opacity-80 flex items-center gap-1">
                              <Shield className="h-3 w-3" />
                              Stop Loss
                            </div>
                            <div className="text-xl font-bold">₹{data[index].final_signal!.stopLoss.toFixed(2)}</div>
                          </div>
                          <div className="bg-black bg-opacity-20 p-3 rounded">
                            <div className="text-sm opacity-80 flex items-center gap-1">
                              <Target className="h-3 w-3" />
                              Target 1
                            </div>
                            <div className="text-xl font-bold">₹{data[index].final_signal!.target1.toFixed(2)}</div>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                          <div className="bg-black bg-opacity-20 p-3 rounded">
                            <div className="text-sm opacity-80">Target 2</div>
                            <div className="text-lg font-bold">₹{data[index].final_signal!.target2.toFixed(2)}</div>
                          </div>
                          <div className="bg-black bg-opacity-20 p-3 rounded">
                            <div className="text-sm opacity-80">Risk:Reward</div>
                            <div className="text-lg font-bold">1:{data[index].final_signal!.riskReward.toFixed(1)}</div>
                          </div>
                          <div className="bg-black bg-opacity-20 p-3 rounded">
                            <div className="text-sm opacity-80">Position Size</div>
                            <div className="text-lg font-bold">{data[index].final_signal!.positionSize}</div>
                          </div>
                        </div>

                        <div className="bg-black bg-opacity-20 p-4 rounded">
                          <div className="text-sm opacity-80 mb-3 font-semibold">Signal Analysis:</div>
                          <ul className="text-sm space-y-2">
                            {data[index].final_signal!.reasoning.map((reason, idx) => (
                              <li key={idx} className="flex items-start gap-2">
                                <div className="w-1.5 h-1.5 bg-white rounded-full mt-2 flex-shrink-0"></div>
                                <span>{reason}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </>
              ) : (
                <Card className="bg-gray-800 border-gray-700">
                  <CardContent className="p-8 text-center">
                    <div className="text-white mb-4">
                      {connectionStatus[index] === "connected" ? (
                        <div className="flex items-center justify-center gap-2">
                          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                          Loading high-accuracy signals with options Greeks...
                        </div>
                      ) : connectionStatus[index] === "error" ? (
                        <div className="flex items-center justify-center gap-2 text-red-400">
                          <AlertTriangle className="h-6 w-6" />
                          Connection error - Start backend server with Angel One API credentials
                        </div>
                      ) : (
                        <div className="flex items-center justify-center gap-2 text-gray-400">
                          <WifiOff className="h-6 w-6" />
                          Connecting to {index} data feed...
                        </div>
                      )}
                    </div>
                    <div className="text-gray-400 text-sm">
                      To see trading signals with options Greeks:
                      <br />
                      1. Configure Angel One API credentials in backend.py
                      <br />
                      2. Install dependencies: pip install -r requirements.txt
                      <br />
                      3. Run: python backend.py
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
          ))}
        </Tabs>

        {/* Enhanced Connection Status */}
        <div className="mt-6 text-center">
          <div className="flex items-center justify-center gap-6 text-sm mb-2">
            {INDICES.map((index) => (
              <div key={index} className="flex items-center gap-2">
                <span className="text-gray-400">{index}:</span>
                <span
                  className={
                    connectionStatus[index] === "connected"
                      ? "text-green-400"
                      : connectionStatus[index] === "error"
                        ? "text-red-400"
                        : "text-gray-400"
                  }
                >
                  {connectionStatus[index]}
                </span>
                {index === activeIndex && connectionStatus[index] === "connected" && (
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                )}
              </div>
            ))}
          </div>
          <p className="text-gray-400 text-xs">
            Real-time data with Options Greeks • Market Status Monitoring • Trading Calendar • 5-second updates
          </p>
        </div>
      </div>
    </div>
  )
}
