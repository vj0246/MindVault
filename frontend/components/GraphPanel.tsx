'use client'

import { useState, useEffect, useRef, useCallback } from 'react'

// ─── Types ────────────────────────────────────────────────────────────────────

interface GraphNode {
  id: string
  sources: string[]
  x: number
  y: number
  vx: number
  vy: number
  connections: number
}

interface GraphEdge {
  source: string
  target: string
  relation: string
}

interface GraphData {
  nodes: { id: string; sources: string[] }[]
  edges: GraphEdge[]
}

// ─── Force simulation (no D3 needed) ─────────────────────────────────────────

function buildSimulation(rawNodes: { id: string; sources: string[] }[], edges: GraphEdge[]) {
  const W = 900, H = 600
  const connectionCount: Record<string, number> = {}

  edges.forEach(e => {
    connectionCount[e.source] = (connectionCount[e.source] || 0) + 1
    connectionCount[e.target] = (connectionCount[e.target] || 0) + 1
  })

  const nodes: GraphNode[] = rawNodes.map((n, i) => ({
    ...n,
    x: W / 2 + (Math.random() - 0.5) * 300,
    y: H / 2 + (Math.random() - 0.5) * 300,
    vx: 0, vy: 0,
    connections: connectionCount[n.id] || 0,
  }))

  // Run 200 iterations of force simulation
  const nodeMap: Record<string, GraphNode> = {}
  nodes.forEach(n => nodeMap[n.id] = n)

  for (let iter = 0; iter < 200; iter++) {
    const alpha = 1 - iter / 200

    // Repulsion between all nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j]
        const dx = b.x - a.x, dy = b.y - a.y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        const force = (5000 / (dist * dist)) * alpha
        const fx = (dx / dist) * force
        const fy = (dy / dist) * force
        a.vx -= fx; a.vy -= fy
        b.vx += fx; b.vy += fy
      }
    }

    // Attraction along edges
    edges.forEach(e => {
      const a = nodeMap[e.source], b = nodeMap[e.target]
      if (!a || !b) return
      const dx = b.x - a.x, dy = b.y - a.y
      const dist = Math.sqrt(dx * dx + dy * dy) || 1
      const target = 120
      const force = ((dist - target) / dist) * 0.05 * alpha
      a.vx += dx * force; a.vy += dy * force
      b.vx -= dx * force; b.vy -= dy * force
    })

    // Center gravity
    nodes.forEach(n => {
      n.vx += (W / 2 - n.x) * 0.01 * alpha
      n.vy += (H / 2 - n.y) * 0.01 * alpha
    })

    // Apply velocity with damping
    nodes.forEach(n => {
      n.vx *= 0.8; n.vy *= 0.8
      n.x += n.vx; n.y += n.vy
      n.x = Math.max(50, Math.min(W - 50, n.x))
      n.y = Math.max(50, Math.min(H - 50, n.y))
    })
  }

  return nodes
}

// ─── Main Graph Panel ─────────────────────────────────────────────────────────

export default function GraphPanel({ open, topic, data, loading, onClose, onNodeClick }: {
  open: boolean
  topic: string
  data: GraphData | null
  loading: boolean
  onClose: () => void
  onNodeClick: (id: string) => void
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [fullscreen, setFullscreen] = useState(false)
  const [search, setSearch] = useState('')
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const [draggingNode, setDraggingNode] = useState<string | null>(null)
  const [isPanning, setIsPanning] = useState(false)
  const panStart = useRef({ x: 0, y: 0, px: 0, py: 0 })
  const dragOffset = useRef({ x: 0, y: 0 })
  const [searchResult, setSearchResult] = useState<string | null>(null)

  const W = 900, H = 600

  // Build simulation when data changes
  useEffect(() => {
    if (!data || data.nodes.length === 0) { setNodes([]); return }
    const simNodes = buildSimulation(data.nodes, data.edges)
    setNodes(simNodes)
    setPan({ x: 0, y: 0 })
    setZoom(1)
    setSelectedNode(null)
    setSearch('')
    setSearchResult(null)
  }, [data])

  // Search handler
  const handleSearch = useCallback(() => {
    if (!search.trim()) { setSearchResult(null); return }
    const q = search.toLowerCase()
    const match = nodes.find(n => n.id.includes(q) || q.includes(n.id))
    if (match) {
      setSearchResult(match.id)
      setSelectedNode(match.id)
      // Pan to center on found node
      const targetX = W / 2 - match.x * zoom
      const targetY = H / 2 - match.y * zoom
      setPan({ x: targetX, y: targetY })
    } else {
      setSearchResult('not_found')
    }
  }, [search, nodes, zoom])

  // Drag node
  const handleNodeMouseDown = (e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation()
    setDraggingNode(nodeId)
    setSelectedNode(nodeId)
    const svgRect = svgRef.current?.getBoundingClientRect()
    if (!svgRect) return
    const node = nodes.find(n => n.id === nodeId)
    if (!node) return
    dragOffset.current = {
      x: (e.clientX - svgRect.left - pan.x) / zoom - node.x,
      y: (e.clientY - svgRect.top - pan.y) / zoom - node.y,
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    const svgRect = svgRef.current?.getBoundingClientRect()
    if (!svgRect) return

    if (draggingNode) {
      const x = (e.clientX - svgRect.left - pan.x) / zoom - dragOffset.current.x
      const y = (e.clientY - svgRect.top - pan.y) / zoom - dragOffset.current.y
      setNodes(prev => prev.map(n => n.id === draggingNode
        ? { ...n, x: Math.max(30, Math.min(W - 30, x)), y: Math.max(30, Math.min(H - 30, y)) }
        : n
      ))
    } else if (isPanning) {
      setPan({
        x: panStart.current.px + (e.clientX - panStart.current.x),
        y: panStart.current.py + (e.clientY - panStart.current.y),
      })
    }
  }

  const handleMouseUp = () => {
    setDraggingNode(null)
    setIsPanning(false)
  }

  const handleSvgMouseDown = (e: React.MouseEvent) => {
    if (e.target === svgRef.current || (e.target as SVGElement).tagName === 'rect') {
      setIsPanning(true)
      panStart.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y }
      setSelectedNode(null)
    }
  }

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    setZoom(z => Math.max(0.3, Math.min(3, z * delta)))
  }

  // Node radius based on connections
  const nodeRadius = (n: GraphNode) => Math.max(18, Math.min(36, 18 + n.connections * 3))

  // Color based on source count
  const nodeColor = (n: GraphNode, isSelected: boolean, isHovered: boolean) => {
    if (isSelected) return { fill: 'rgba(232,197,71,0.25)', stroke: 'rgba(232,197,71,0.9)', glow: true }
    if (isHovered) return { fill: 'rgba(232,197,71,0.15)', stroke: 'rgba(232,197,71,0.6)', glow: false }
    if (n.sources.length > 1) return { fill: 'rgba(196,149,106,0.15)', stroke: 'rgba(196,149,106,0.5)', glow: false }
    if (n.connections > 3) return { fill: 'rgba(126,184,164,0.15)', stroke: 'rgba(126,184,164,0.5)', glow: false }
    return { fill: 'rgba(100,120,150,0.1)', stroke: 'rgba(100,120,150,0.35)', glow: false }
  }

  // Get connected nodes for selected
  const getConnected = (nodeId: string) => {
    if (!data) return new Set<string>()
    const connected = new Set<string>([nodeId])
    data.edges.forEach(e => {
      if (e.source === nodeId) connected.add(e.target)
      if (e.target === nodeId) connected.add(e.source)
    })
    return connected
  }

  const connectedNodes = selectedNode ? getConnected(selectedNode) : null

  const panelWidth = fullscreen ? '100vw' : 560
  const panelHeight = fullscreen ? '100vh' : '100vh'

  if (!open) return null

  return (
    <div style={{
      position: 'fixed',
      top: 0, right: 0,
      width: panelWidth,
      height: panelHeight,
      background: 'var(--surface)',
      borderLeft: '1px solid var(--border)',
      zIndex: 60,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      transition: 'width 0.3s ease',
    }}>

      {/* Header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        flexShrink: 0,
      }}>
        <div style={{ flex: 1 }}>
          <p style={{ fontFamily: 'Instrument Serif', fontSize: 16, fontStyle: 'italic', color: 'var(--text)', lineHeight: 1 }}>
            Knowledge Graph
          </p>
          {data && (
            <p style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', marginTop: 2 }}>
              {data.nodes.length} nodes · {data.edges.length} edges · topic: {topic}
            </p>
          )}
        </div>

        {/* Controls */}
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button
            onClick={() => setZoom(z => Math.min(3, z * 1.2))}
            style={{ ...iconBtn, fontSize: 14 }} title="Zoom in"
          >+</button>
          <button
            onClick={() => setZoom(z => Math.max(0.3, z * 0.8))}
            style={{ ...iconBtn, fontSize: 14 }} title="Zoom out"
          >−</button>
          <button
            onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }) }}
            style={iconBtn} title="Reset view"
          >⌂</button>
          <button
            onClick={() => setFullscreen(f => !f)}
            style={iconBtn} title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          >{fullscreen ? '⊡' : '⊞'}</button>
          <button onClick={onClose} style={{ ...iconBtn, color: 'var(--danger)' }} title="Close">✕</button>
        </div>
      </div>

      {/* Search bar */}
      <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            value={search}
            onChange={e => { setSearch(e.target.value); setSearchResult(null) }}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
            placeholder="Search a concept... (e.g. deadlock, mutex, scheduler)"
            style={{
              flex: 1,
              background: 'var(--bg)',
              border: '1px solid var(--border2)',
              borderRadius: 6,
              padding: '7px 12px',
              fontSize: 12,
              color: 'var(--text)',
              fontFamily: 'IBM Plex Mono',
              outline: 'none',
            }}
          />
          <button
            onClick={handleSearch}
            style={{
              background: 'var(--accent)',
              color: 'var(--bg)',
              border: 'none',
              borderRadius: 6,
              padding: '7px 14px',
              fontSize: 11,
              fontFamily: 'IBM Plex Mono',
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >Find</button>
        </div>
        {searchResult === 'not_found' && (
          <p style={{ fontSize: 10, color: 'var(--danger)', fontFamily: 'IBM Plex Mono', marginTop: 5 }}>
            No matching node found. Try a shorter keyword.
          </p>
        )}
        {searchResult && searchResult !== 'not_found' && (
          <p style={{ fontSize: 10, color: 'var(--accent3)', fontFamily: 'IBM Plex Mono', marginTop: 5 }}>
            Found: "{searchResult}" — showing connected nodes
          </p>
        )}
      </div>

      {/* Graph area */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden', background: 'var(--bg)' }}>
        {loading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 10 }}>
            <div className="spinner" />
            <span style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>Building graph...</span>
          </div>
        ) : !data || nodes.length === 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 12 }}>
            <p style={{ fontSize: 28 }}>🕸️</p>
            <p style={{ fontSize: 13, color: 'var(--text2)' }}>No graph data for "{topic}"</p>
            <p style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>
              Upload documents first to build the knowledge graph.
            </p>
          </div>
        ) : (
          <>
            {/* Hint overlay */}
            <div style={{
              position: 'absolute', top: 10, left: 12, zIndex: 10,
              fontSize: 9, color: 'var(--text3)', fontFamily: 'IBM Plex Mono',
              display: 'flex', flexDirection: 'column', gap: 2,
              pointerEvents: 'none',
            }}>
              <span>Drag nodes · Scroll to zoom · Click node to select</span>
              <span style={{ color: 'rgba(232,197,71,0.5)' }}>
                {selectedNode ? `Selected: ${selectedNode}` : 'Click a node to see connections'}
              </span>
            </div>

            {/* Zoom indicator */}
            <div style={{
              position: 'absolute', bottom: 10, right: 12, zIndex: 10,
              fontSize: 9, color: 'var(--text3)', fontFamily: 'IBM Plex Mono',
            }}>
              {Math.round(zoom * 100)}%
            </div>

            <svg
              ref={svgRef}
              width="100%"
              height="100%"
              viewBox={`0 0 ${W} ${H}`}
              style={{ cursor: draggingNode ? 'grabbing' : isPanning ? 'grabbing' : 'grab', display: 'block' }}
              onMouseDown={handleSvgMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              onWheel={handleWheel}
              preserveAspectRatio="xMidYMid meet"
            >
              {/* Defs for glow */}
              <defs>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                  <feMerge>
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <filter id="softglow">
                  <feGaussianBlur stdDeviation="1.5" result="coloredBlur" />
                  <feMerge>
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                  <path d="M 0 0 L 6 3 L 0 6 z" fill="rgba(232,197,71,0.3)" />
                </marker>
              </defs>

              {/* Background grid */}
              <rect width={W} height={H} fill="transparent" />
              <g opacity="0.04">
                {Array.from({ length: 20 }).map((_, i) => (
                  <line key={`v${i}`} x1={i * 50} y1={0} x2={i * 50} y2={H} stroke="var(--accent)" strokeWidth={0.5} />
                ))}
                {Array.from({ length: 13 }).map((_, i) => (
                  <line key={`h${i}`} x1={0} y1={i * 50} x2={W} y2={i * 50} stroke="var(--accent)" strokeWidth={0.5} />
                ))}
              </g>

              <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
                {/* Edges */}
                {data.edges.map((edge, i) => {
                  const from = nodes.find(n => n.id === edge.source)
                  const to = nodes.find(n => n.id === edge.target)
                  if (!from || !to) return null

                  const isHighlighted = connectedNodes
                    ? connectedNodes.has(edge.source) && connectedNodes.has(edge.target)
                    : true

                  const midX = (from.x + to.x) / 2
                  const midY = (from.y + to.y) / 2

                  // Offset midpoint slightly for curve
                  const dx = to.x - from.x, dy = to.y - from.y
                  const len = Math.sqrt(dx * dx + dy * dy) || 1
                  const mx = midX - (dy / len) * 20
                  const my = midY + (dx / len) * 20

                  return (
                    <g key={i} opacity={isHighlighted ? 1 : 0.12}>
                      <path
                        d={`M ${from.x} ${from.y} Q ${mx} ${my} ${to.x} ${to.y}`}
                        fill="none"
                        stroke={isHighlighted ? 'rgba(232,197,71,0.35)' : 'rgba(232,197,71,0.15)'}
                        strokeWidth={isHighlighted ? 1.5 : 1}
                        markerEnd="url(#arrow)"
                      />
                      {isHighlighted && (
                        <text x={mx} y={my - 6} textAnchor="middle"
                          style={{ fontSize: 8, fill: 'rgba(232,197,71,0.6)', fontFamily: 'IBM Plex Mono', pointerEvents: 'none' }}>
                          {edge.relation.length > 18 ? edge.relation.slice(0, 17) + '…' : edge.relation}
                        </text>
                      )}
                    </g>
                  )
                })}

                {/* Nodes */}
                {nodes.map((node) => {
                  const isSelected = selectedNode === node.id
                  const isHovered = hoveredNode === node.id
                  const isConnected = connectedNodes ? connectedNodes.has(node.id) : true
                  const isSearch = searchResult === node.id
                  const r = nodeRadius(node)
                  const colors = nodeColor(node, isSelected || isSearch, isHovered)

                  return (
                    <g
                      key={node.id}
                      style={{ cursor: 'pointer' }}
                      onMouseDown={(e) => handleNodeMouseDown(e, node.id)}
                      onMouseEnter={() => setHoveredNode(node.id)}
                      onMouseLeave={() => setHoveredNode(null)}
                      onDoubleClick={() => onNodeClick(node.id)}
                      opacity={!isConnected ? 0.2 : 1}
                    >
                      {/* Glow ring for selected/search */}
                      {(isSelected || isSearch) && (
                        <circle cx={node.x} cy={node.y} r={r + 8}
                          fill="none"
                          stroke={isSearch ? 'rgba(126,184,164,0.4)' : 'rgba(232,197,71,0.2)'}
                          strokeWidth={1}
                          strokeDasharray="4 3"
                          filter="url(#softglow)"
                        />
                      )}

                      {/* Node circle */}
                      <circle
                        cx={node.x} cy={node.y} r={r}
                        fill={colors.fill}
                        stroke={colors.stroke}
                        strokeWidth={isSelected ? 2 : 1}
                        filter={colors.glow ? 'url(#glow)' : undefined}
                      />

                      {/* Connection count badge */}
                      {node.connections > 0 && (
                        <circle cx={node.x + r * 0.7} cy={node.y - r * 0.7} r={6}
                          fill="rgba(232,197,71,0.15)" stroke="rgba(232,197,71,0.4)" strokeWidth={0.5} />
                      )}
                      {node.connections > 0 && (
                        <text x={node.x + r * 0.7} y={node.y - r * 0.7}
                          textAnchor="middle" dominantBaseline="middle"
                          style={{ fontSize: 6, fill: 'rgba(232,197,71,0.8)', fontFamily: 'IBM Plex Mono', pointerEvents: 'none' }}>
                          {node.connections}
                        </text>
                      )}

                      {/* Node label */}
                      <text
                        x={node.x} y={node.y}
                        textAnchor="middle" dominantBaseline="middle"
                        style={{
                          fontSize: Math.max(7, Math.min(10, 100 / node.id.length)),
                          fill: isSelected || isSearch ? 'var(--accent)' : isHovered ? 'var(--text)' : 'var(--text2)',
                          fontFamily: 'IBM Plex Mono',
                          pointerEvents: 'none',
                          fontWeight: isSelected ? '600' : '400',
                        }}
                      >
                        {node.id.length > 14 ? node.id.slice(0, 13) + '…' : node.id}
                      </text>

                      {/* Source indicator */}
                      {node.sources.length > 1 && (
                        <text x={node.x} y={node.y + r + 10}
                          textAnchor="middle"
                          style={{ fontSize: 7, fill: 'rgba(196,149,106,0.6)', fontFamily: 'IBM Plex Mono', pointerEvents: 'none' }}>
                          {node.sources.length} sources
                        </text>
                      )}
                    </g>
                  )
                })}
              </g>
            </svg>
          </>
        )}
      </div>

      {/* Bottom panel — selected node info */}
      {selectedNode && data && (
        <div style={{
          borderTop: '1px solid var(--border)',
          padding: '12px 16px',
          flexShrink: 0,
          background: 'rgba(232,197,71,0.04)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <p style={{ fontSize: 13, color: 'var(--accent)', fontFamily: 'IBM Plex Mono', fontWeight: 600 }}>
              {selectedNode}
            </p>
            <button
              onClick={() => onNodeClick(selectedNode)}
              style={{
                background: 'var(--accent)',
                color: 'var(--bg)',
                border: 'none',
                borderRadius: 5,
                padding: '5px 12px',
                fontSize: 10,
                fontFamily: 'IBM Plex Mono',
                cursor: 'pointer',
                fontWeight: 600,
              }}
            >
              Ask MindVault →
            </button>
          </div>

          {/* Edges involving selected node */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 80, overflowY: 'auto' }}>
            {data.edges
              .filter(e => e.source === selectedNode || e.target === selectedNode)
              .slice(0, 5)
              .map((e, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10, fontFamily: 'IBM Plex Mono' }}>
                  <span style={{ color: 'var(--accent3)' }}>{e.source}</span>
                  <span style={{ color: 'var(--text3)' }}>──[{e.relation}]──▶</span>
                  <span style={{ color: 'var(--accent2)' }}>{e.target}</span>
                </div>
              ))
            }
          </div>

          <p style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', marginTop: 6 }}>
            Double-click node to query · Drag to reposition
          </p>
        </div>
      )}

      {/* Legend */}
      <div style={{
        borderTop: '1px solid var(--border)',
        padding: '8px 16px',
        display: 'flex',
        gap: 16,
        alignItems: 'center',
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Legend</span>
        <div style={{ display: 'flex', gap: 12 }}>
          {[
            { color: 'rgba(126,184,164,0.5)', label: 'Hub node' },
            { color: 'rgba(196,149,106,0.5)', label: 'Multi-source' },
            { color: 'rgba(100,120,150,0.35)', label: 'Standard' },
            { color: 'rgba(232,197,71,0.9)', label: 'Selected' },
          ].map((l, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: l.color, border: `1px solid ${l.color}` }} />
              <span style={{ fontSize: 9, color: 'var(--text3)', fontFamily: 'IBM Plex Mono' }}>{l.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── Shared button style ──────────────────────────────────────────────────────

const iconBtn: React.CSSProperties = {
  background: 'var(--surface2)',
  border: '1px solid var(--border2)',
  color: 'var(--text2)',
  borderRadius: 5,
  width: 28,
  height: 28,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  cursor: 'pointer',
  fontSize: 12,
  fontFamily: 'IBM Plex Mono',
  flexShrink: 0,
}
