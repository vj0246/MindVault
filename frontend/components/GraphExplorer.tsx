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
  radius: number
  pinned: boolean
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

interface Props {
  open: boolean
  initialTopic: string
  onClose: () => void
  onNodeQuery: (id: string) => void
  fetchGraph: (topic: string) => Promise<GraphData>
}

// ─── Physics constants ────────────────────────────────────────────────────────

const REPULSION = 4000
const ATTRACTION = 0.04
const DAMPING = 0.85
const MIN_DIST = 60
const CENTER_PULL = 0.008

export default function GraphExplorer({ open, initialTopic, onClose, onNodeQuery, fetchGraph }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number>(0)
  const nodesRef = useRef<GraphNode[]>([])
  const edgesRef = useRef<GraphEdge[]>([])
  const dragRef = useRef<{ node: GraphNode | null; offsetX: number; offsetY: number }>({ node: null, offsetX: 0, offsetY: 0 })
  const hoverRef = useRef<GraphNode | null>(null)
  const fullscreenRef = useRef(false)

  const [isFullscreen, setIsFullscreen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [searchInput, setSearchInput] = useState('')
  const [topic, setTopic] = useState(initialTopic)
  const [nodeCount, setNodeCount] = useState(0)
  const [edgeCount, setEdgeCount] = useState(0)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [history, setHistory] = useState<string[]>([])
  const containerRef = useRef<HTMLDivElement>(null)

  // ─── Load graph data ────────────────────────────────────────────────────────

  const loadGraph = useCallback(async (t: string) => {
    setLoading(true)
    setSelectedNode(null)
    try {
      const data = await fetchGraph(t)
      if (!data || data.nodes.length === 0) {
        nodesRef.current = []
        edgesRef.current = []
        setNodeCount(0)
        setEdgeCount(0)
        return
      }

      const canvas = canvasRef.current
      const W = canvas?.width || 800
      const H = canvas?.height || 600
      const cx = W / 2, cy = H / 2

      // Place nodes in a circle initially
      const newNodes: GraphNode[] = data.nodes.map((n, i) => {
        const angle = (i / data.nodes.length) * 2 * Math.PI
        const r = Math.min(W, H) * 0.3
        const isTopic = n.id === t.toLowerCase() || n.id.includes(t.toLowerCase())
        return {
          id: n.id,
          sources: n.sources,
          x: isTopic ? cx : cx + r * Math.cos(angle) + (Math.random() - 0.5) * 40,
          y: isTopic ? cy : cy + r * Math.sin(angle) + (Math.random() - 0.5) * 40,
          vx: 0, vy: 0,
          radius: isTopic ? 30 : 20,
          pinned: false,
        }
      })

      nodesRef.current = newNodes
      edgesRef.current = data.edges
      setNodeCount(data.nodes.length)
      setEdgeCount(data.edges.length)
      setHistory(prev => prev.includes(t) ? prev : [...prev.slice(-9), t])
    } catch (e) {
      console.error('Graph load failed', e)
    } finally {
      setLoading(false)
    }
  }, [fetchGraph])

  useEffect(() => {
    if (open && initialTopic) {
      setTopic(initialTopic)
      loadGraph(initialTopic)
    }
  }, [open, initialTopic, loadGraph])

  // ─── Physics simulation ─────────────────────────────────────────────────────

  const simulate = useCallback(() => {
    const nodes = nodesRef.current
    const edges = edgesRef.current
    const canvas = canvasRef.current
    if (!canvas || nodes.length === 0) return

    const W = canvas.width
    const H = canvas.height
    const cx = W / 2, cy = H / 2

    // Repulsion between all nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x
        const dy = nodes[i].y - nodes[j].y
        const dist = Math.sqrt(dx * dx + dy * dy) || 1
        if (dist < MIN_DIST * 3) {
          const force = REPULSION / (dist * dist)
          const fx = (dx / dist) * force
          const fy = (dy / dist) * force
          if (!nodes[i].pinned) { nodes[i].vx += fx; nodes[i].vy += fy }
          if (!nodes[j].pinned) { nodes[j].vx -= fx; nodes[j].vy -= fy }
        }
      }
    }

    // Attraction along edges
    for (const edge of edges) {
      const s = nodes.find(n => n.id === edge.source)
      const t = nodes.find(n => n.id === edge.target)
      if (!s || !t) continue
      const dx = t.x - s.x
      const dy = t.y - s.y
      const dist = Math.sqrt(dx * dx + dy * dy) || 1
      const ideal = 120
      const force = (dist - ideal) * ATTRACTION
      const fx = (dx / dist) * force
      const fy = (dy / dist) * force
      if (!s.pinned) { s.vx += fx; s.vy += fy }
      if (!t.pinned) { t.vx -= fx; t.vy -= fy }
    }

    // Center pull and update positions
    for (const node of nodes) {
      if (node.pinned) continue
      node.vx += (cx - node.x) * CENTER_PULL
      node.vy += (cy - node.y) * CENTER_PULL
      node.vx *= DAMPING
      node.vy *= DAMPING
      node.x += node.vx
      node.y += node.vy
      // Boundary
      node.x = Math.max(node.radius + 10, Math.min(W - node.radius - 10, node.x))
      node.y = Math.max(node.radius + 10, Math.min(H - node.radius - 10, node.y))
    }
  }, [])

  // ─── Drawing ────────────────────────────────────────────────────────────────

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height
    const nodes = nodesRef.current
    const edges = edgesRef.current
    const hovered = hoverRef.current

    // Clear
    ctx.fillStyle = '#0a0a0a'
    ctx.fillRect(0, 0, W, H)

    // Grid
    ctx.strokeStyle = 'rgba(232,197,71,0.04)'
    ctx.lineWidth = 1
    for (let x = 0; x < W; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
    }
    for (let y = 0; y < H; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
    }

    if (nodes.length === 0) {
      ctx.fillStyle = 'rgba(255,255,255,0.2)'
      ctx.font = '14px IBM Plex Mono'
      ctx.textAlign = 'center'
      ctx.fillText('No graph data. Enter a topic above.', W / 2, H / 2)
      return
    }

    // Draw edges
    for (const edge of edges) {
      const s = nodes.find(n => n.id === edge.source)
      const t = nodes.find(n => n.id === edge.target)
      if (!s || !t) continue

      const isHighlighted = hovered && (hovered.id === edge.source || hovered.id === edge.target)
      const isSelected = selectedNode && (selectedNode === edge.source || selectedNode === edge.target)

      // Edge line
      const grad = ctx.createLinearGradient(s.x, s.y, t.x, t.y)
      if (isHighlighted || isSelected) {
        grad.addColorStop(0, 'rgba(232,197,71,0.6)')
        grad.addColorStop(1, 'rgba(126,184,164,0.6)')
      } else {
        grad.addColorStop(0, 'rgba(232,197,71,0.12)')
        grad.addColorStop(1, 'rgba(126,184,164,0.12)')
      }

      ctx.strokeStyle = grad
      ctx.lineWidth = isHighlighted || isSelected ? 2 : 1
      ctx.setLineDash(isHighlighted || isSelected ? [] : [4, 4])
      ctx.beginPath()
      ctx.moveTo(s.x, s.y)
      ctx.lineTo(t.x, t.y)
      ctx.stroke()
      ctx.setLineDash([])

      // Arrow
      if (isHighlighted || isSelected) {
        const angle = Math.atan2(t.y - s.y, t.x - s.x)
        const arrowX = t.x - Math.cos(angle) * (t.radius + 4)
        const arrowY = t.y - Math.sin(angle) * (t.radius + 4)
        ctx.fillStyle = 'rgba(232,197,71,0.6)'
        ctx.beginPath()
        ctx.moveTo(arrowX, arrowY)
        ctx.lineTo(arrowX - 8 * Math.cos(angle - 0.4), arrowY - 8 * Math.sin(angle - 0.4))
        ctx.lineTo(arrowX - 8 * Math.cos(angle + 0.4), arrowY - 8 * Math.sin(angle + 0.4))
        ctx.closePath()
        ctx.fill()
      }

      // Relation label
      if (isHighlighted || isSelected) {
        const mx = (s.x + t.x) / 2
        const my = (s.y + t.y) / 2
        const label = edge.relation.length > 20 ? edge.relation.slice(0, 18) + '…' : edge.relation
        ctx.font = '9px IBM Plex Mono'
        ctx.textAlign = 'center'
        const tw = ctx.measureText(label).width
        ctx.fillStyle = 'rgba(10,10,10,0.85)'
        ctx.fillRect(mx - tw / 2 - 4, my - 10, tw + 8, 16)
        ctx.fillStyle = 'rgba(232,197,71,0.8)'
        ctx.fillText(label, mx, my + 1)
      }
    }

    // Draw nodes
    for (const node of nodes) {
      const isHovered = hovered?.id === node.id
      const isSelected = selectedNode === node.id
      const isTopic = node.id === topic.toLowerCase() || node.id.includes(topic.toLowerCase())

      // Glow
      if (isHovered || isSelected || isTopic) {
        const glowColor = isTopic ? 'rgba(232,197,71,0.15)' : 'rgba(126,184,164,0.15)'
        const glow = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius * 2.5)
        glow.addColorStop(0, glowColor)
        glow.addColorStop(1, 'transparent')
        ctx.fillStyle = glow
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius * 2.5, 0, Math.PI * 2)
        ctx.fill()
      }

      // Outer ring for topic node
      if (isTopic) {
        ctx.strokeStyle = 'rgba(232,197,71,0.3)'
        ctx.lineWidth = 1
        ctx.setLineDash([3, 3])
        ctx.beginPath()
        ctx.arc(node.x, node.y, node.radius + 8, 0, Math.PI * 2)
        ctx.stroke()
        ctx.setLineDash([])
      }

      // Node circle
      const nodeGrad = ctx.createRadialGradient(node.x - node.radius * 0.3, node.y - node.radius * 0.3, 0, node.x, node.y, node.radius)
      if (isTopic) {
        nodeGrad.addColorStop(0, 'rgba(232,197,71,0.35)')
        nodeGrad.addColorStop(1, 'rgba(232,197,71,0.15)')
      } else if (isSelected) {
        nodeGrad.addColorStop(0, 'rgba(126,184,164,0.4)')
        nodeGrad.addColorStop(1, 'rgba(126,184,164,0.15)')
      } else {
        nodeGrad.addColorStop(0, 'rgba(126,184,164,0.2)')
        nodeGrad.addColorStop(1, 'rgba(126,184,164,0.06)')
      }

      ctx.fillStyle = nodeGrad
      ctx.beginPath()
      ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
      ctx.fill()

      // Border
      ctx.strokeStyle = isTopic ? 'rgba(232,197,71,0.7)' : isSelected ? 'rgba(126,184,164,0.7)' : isHovered ? 'rgba(126,184,164,0.5)' : 'rgba(126,184,164,0.2)'
      ctx.lineWidth = isTopic ? 1.5 : 1
      ctx.beginPath()
      ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
      ctx.stroke()

      // Label inside
      const maxLen = node.radius > 22 ? 12 : 8
      const label = node.id.length > maxLen ? node.id.slice(0, maxLen - 1) + '…' : node.id
      ctx.font = `${isTopic ? 9 : 8}px IBM Plex Mono`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillStyle = isTopic ? 'rgba(232,197,71,0.95)' : 'rgba(126,184,164,0.85)'
      ctx.fillText(label, node.x, node.y)

      // Label below on hover
      if ((isHovered || isSelected) && node.id.length > maxLen) {
        ctx.font = '9px IBM Plex Mono'
        ctx.textBaseline = 'top'
        const fullLabel = node.id
        const tw = ctx.measureText(fullLabel).width
        ctx.fillStyle = 'rgba(10,10,10,0.9)'
        ctx.fillRect(node.x - tw / 2 - 5, node.y + node.radius + 4, tw + 10, 16)
        ctx.fillStyle = isTopic ? 'rgba(232,197,71,0.95)' : 'rgba(126,184,164,0.9)'
        ctx.fillText(fullLabel, node.x, node.y + node.radius + 6)
      }

      // Pin indicator
      if (node.pinned) {
        ctx.fillStyle = 'rgba(232,197,71,0.8)'
        ctx.beginPath()
        ctx.arc(node.x + node.radius - 4, node.y - node.radius + 4, 3, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }, [topic, selectedNode])

  // ─── Animation loop ─────────────────────────────────────────────────────────

  useEffect(() => {
    if (!open) return
    const loop = () => {
      simulate()
      draw()
      animRef.current = requestAnimationFrame(loop)
    }
    animRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(animRef.current)
  }, [open, simulate, draw])

  // ─── Canvas resize ──────────────────────────────────────────────────────────

  useEffect(() => {
    const resize = () => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container) return
      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
    }
    resize()
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
  }, [isFullscreen, open])

  // ─── Mouse events ───────────────────────────────────────────────────────────

  const getNodeAt = (x: number, y: number): GraphNode | null => {
    for (const node of [...nodesRef.current].reverse()) {
      const dx = x - node.x, dy = y - node.y
      if (Math.sqrt(dx * dx + dy * dy) < node.radius + 4) return node
    }
    return null
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    if (dragRef.current.node) {
      dragRef.current.node.x = x - dragRef.current.offsetX
      dragRef.current.node.y = y - dragRef.current.offsetY
      dragRef.current.node.vx = 0
      dragRef.current.node.vy = 0
      return
    }

    const node = getNodeAt(x, y)
    hoverRef.current = node
    setHoveredNode(node?.id || null)
    if (canvasRef.current) canvasRef.current.style.cursor = node ? 'pointer' : 'default'
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const node = getNodeAt(x, y)
    if (node) {
      dragRef.current = { node, offsetX: x - node.x, offsetY: y - node.y }
      node.pinned = true
    }
  }

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const node = getNodeAt(x, y)

    if (dragRef.current.node && dragRef.current.node === node) {
      // It was a click not a drag
      setSelectedNode(prev => prev === node.id ? null : node.id)
    }

    if (dragRef.current.node) {
      // Keep pinned if it was moved significantly
      const moved = Math.abs(dragRef.current.node.x - (x - dragRef.current.offsetX)) > 5
      if (!moved) dragRef.current.node.pinned = false
    }

    dragRef.current = { node: null, offsetX: 0, offsetY: 0 }
  }

  const handleDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const node = getNodeAt(e.clientX - rect.left, e.clientY - rect.top)
    if (node) {
      // Double click = load graph for that node
      setTopic(node.id)
      loadGraph(node.id)
    }
  }

  // ─── Search ──────────────────────────────────────────────────────────────────

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!searchInput.trim()) return
    const t = searchInput.trim().toLowerCase()
    setTopic(t)
    setSearchInput('')
    loadGraph(t)
  }

  // ─── Fullscreen ──────────────────────────────────────────────────────────────

  const toggleFullscreen = () => {
    setIsFullscreen(f => !f)
    setTimeout(() => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container) return
      canvas.width = container.clientWidth
      canvas.height = container.clientHeight
    }, 50)
  }

  // ─── Pin all / Release all ───────────────────────────────────────────────────

  const releaseAll = () => {
    nodesRef.current.forEach(n => { n.pinned = false })
  }

  const centerAll = () => {
    const canvas = canvasRef.current
    if (!canvas) return
    const cx = canvas.width / 2, cy = canvas.height / 2
    const nodes = nodesRef.current
    if (nodes.length === 0) return
    const avgX = nodes.reduce((s, n) => s + n.x, 0) / nodes.length
    const avgY = nodes.reduce((s, n) => s + n.y, 0) / nodes.length
    const dx = cx - avgX, dy = cy - avgY
    nodes.forEach(n => { n.x += dx; n.y += dy; n.vx = 0; n.vy = 0 })
  }

  if (!open) return null

  const panelStyle: React.CSSProperties = isFullscreen
    ? { position: 'fixed', inset: 0, zIndex: 100, display: 'flex', flexDirection: 'column', background: '#0a0a0a' }
    : { position: 'fixed', top: 0, right: 0, width: 560, height: '100vh', zIndex: 50, display: 'flex', flexDirection: 'column', background: '#0a0a0a', borderLeft: '1px solid rgba(232,197,71,0.15)' }

  return (
    <div style={panelStyle} className="fade-up">
      {/* Header */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
        <div style={{ flex: 1 }}>
          <div className="flex items-center gap-2">
            <span style={{ fontSize: 14, fontFamily: 'Instrument Serif', fontStyle: 'italic', color: 'var(--text)' }}>
              Knowledge Graph
            </span>
            <span style={{ fontSize: 9, fontFamily: 'IBM Plex Mono', color: 'rgba(232,197,71,0.6)', background: 'rgba(232,197,71,0.08)', border: '1px solid rgba(232,197,71,0.2)', padding: '1px 6px', borderRadius: 3 }}>
              {topic}
            </span>
            {loading && (
              <div className="spinner" style={{ width: 12, height: 12 }} />
            )}
          </div>
          <p style={{ fontSize: 9, fontFamily: 'IBM Plex Mono', color: 'var(--text3)', marginTop: 2 }}>
            {nodeCount} nodes · {edgeCount} edges · drag to move · dbl-click to explore · click to select
          </p>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={releaseAll} title="Release all pins"
            style={{ padding: '4px 8px', borderRadius: 4, border: '1px solid rgba(255,255,255,0.08)', background: 'none', color: 'var(--text3)', fontSize: 10, fontFamily: 'IBM Plex Mono', cursor: 'pointer' }}>
            unpin
          </button>
          <button onClick={centerAll} title="Center graph"
            style={{ padding: '4px 8px', borderRadius: 4, border: '1px solid rgba(255,255,255,0.08)', background: 'none', color: 'var(--text3)', fontSize: 10, fontFamily: 'IBM Plex Mono', cursor: 'pointer' }}>
            center
          </button>
          <button onClick={toggleFullscreen} title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            style={{ padding: '4px 8px', borderRadius: 4, border: '1px solid rgba(255,255,255,0.08)', background: 'none', color: 'var(--accent3)', fontSize: 10, fontFamily: 'IBM Plex Mono', cursor: 'pointer' }}>
            {isFullscreen ? '⊠ exit' : '⊞ full'}
          </button>
          <button onClick={onClose}
            style={{ padding: '4px 8px', borderRadius: 4, border: '1px solid rgba(255,255,255,0.08)', background: 'none', color: 'var(--text3)', fontSize: 14, cursor: 'pointer', lineHeight: 1 }}>
            ✕
          </button>
        </div>
      </div>

      {/* Search bar */}
      <form onSubmit={handleSearch} style={{ padding: '10px 16px', borderBottom: '1px solid rgba(255,255,255,0.04)', flexShrink: 0, display: 'flex', gap: 8 }}>
        <input
          value={searchInput}
          onChange={e => setSearchInput(e.target.value)}
          placeholder="Enter topic to explore graph... (e.g. deadlock, scheduling, memory)"
          style={{
            flex: 1, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 6, padding: '7px 12px', color: 'var(--text)', fontSize: 12,
            fontFamily: 'IBM Plex Mono', outline: 'none',
          }}
          onFocus={e => e.target.style.borderColor = 'rgba(232,197,71,0.4)'}
          onBlur={e => e.target.style.borderColor = 'rgba(255,255,255,0.08)'}
        />
        <button type="submit"
          style={{ padding: '7px 14px', borderRadius: 6, background: 'rgba(232,197,71,0.15)', border: '1px solid rgba(232,197,71,0.3)', color: 'var(--accent)', fontSize: 12, fontFamily: 'IBM Plex Mono', cursor: 'pointer' }}>
          explore →
        </button>
      </form>

      {/* Canvas */}
      <div ref={containerRef} style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <canvas
          ref={canvasRef}
          style={{ display: 'block', width: '100%', height: '100%' }}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onDoubleClick={handleDoubleClick}
          onMouseLeave={() => { hoverRef.current = null; setHoveredNode(null) }}
        />

        {/* Node detail panel */}
        {selectedNode && (
          <div className="fade-up" style={{
            position: 'absolute', bottom: 16, left: 16, right: 16,
            background: 'rgba(13,13,13,0.95)', border: '1px solid rgba(232,197,71,0.2)',
            borderRadius: 8, padding: '12px 14px', backdropFilter: 'blur(10px)',
          }}>
            <div className="flex items-center justify-between mb-2">
              <div>
                <span style={{ fontSize: 13, color: 'var(--accent)', fontFamily: 'IBM Plex Mono' }}>{selectedNode}</span>
                <span style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', marginLeft: 8 }}>
                  {nodesRef.current.find(n => n.id === selectedNode)?.sources.join(', ')}
                </span>
              </div>
              <button onClick={() => setSelectedNode(null)}
                style={{ color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', fontSize: 12 }}>✕</button>
            </div>
            <div style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'IBM Plex Mono', marginBottom: 8 }}>
              {edgesRef.current.filter(e => e.source === selectedNode || e.target === selectedNode).map((e, i) => (
                <div key={i} style={{ marginBottom: 3 }}>
                  <span style={{ color: 'var(--accent3)' }}>{e.source}</span>
                  <span style={{ color: 'var(--text3)', margin: '0 6px' }}>──[{e.relation}]──▶</span>
                  <span style={{ color: 'var(--accent2)' }}>{e.target}</span>
                </div>
              ))}
            </div>
            <div className="flex gap-2">
              <button onClick={() => { setTopic(selectedNode); loadGraph(selectedNode) }}
                style={{ padding: '5px 10px', borderRadius: 5, background: 'rgba(126,184,164,0.1)', border: '1px solid rgba(126,184,164,0.25)', color: 'var(--accent3)', fontSize: 10, fontFamily: 'IBM Plex Mono', cursor: 'pointer' }}>
                ⬡ expand graph
              </button>
              <button onClick={() => { onNodeQuery(selectedNode); setSelectedNode(null) }}
                style={{ padding: '5px 10px', borderRadius: 5, background: 'rgba(232,197,71,0.1)', border: '1px solid rgba(232,197,71,0.25)', color: 'var(--accent)', fontSize: 10, fontFamily: 'IBM Plex Mono', cursor: 'pointer' }}>
                → query this
              </button>
            </div>
          </div>
        )}

        {/* History breadcrumb */}
        {history.length > 1 && (
          <div style={{ position: 'absolute', top: 10, left: 10, display: 'flex', gap: 4, flexWrap: 'wrap', maxWidth: '60%' }}>
            {history.map((h, i) => (
              <button key={i} onClick={() => { setTopic(h); loadGraph(h) }}
                style={{
                  padding: '2px 7px', borderRadius: 3, fontSize: 9, fontFamily: 'IBM Plex Mono', cursor: 'pointer',
                  background: h === topic ? 'rgba(232,197,71,0.15)' : 'rgba(0,0,0,0.6)',
                  border: `1px solid ${h === topic ? 'rgba(232,197,71,0.3)' : 'rgba(255,255,255,0.08)'}`,
                  color: h === topic ? 'var(--accent)' : 'var(--text3)',
                }}>
                {h}
              </button>
            ))}
          </div>
        )}

        {/* Legend */}
        <div style={{ position: 'absolute', top: 10, right: 10, background: 'rgba(0,0,0,0.7)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 6, padding: '8px 10px', fontSize: 9, fontFamily: 'IBM Plex Mono' }}>
          <div className="flex items-center gap-2 mb-1">
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: 'rgba(232,197,71,0.25)', border: '1px solid rgba(232,197,71,0.7)' }} />
            <span style={{ color: 'var(--text3)' }}>topic node</span>
          </div>
          <div className="flex items-center gap-2 mb-1">
            <div style={{ width: 10, height: 10, borderRadius: '50%', background: 'rgba(126,184,164,0.15)', border: '1px solid rgba(126,184,164,0.3)' }} />
            <span style={{ color: 'var(--text3)' }}>related node</span>
          </div>
          <div className="flex items-center gap-2">
            <div style={{ width: 10, height: 1, background: 'rgba(232,197,71,0.3)', borderTop: '1px dashed rgba(232,197,71,0.3)' }} />
            <span style={{ color: 'var(--text3)' }}>relationship</span>
          </div>
        </div>
      </div>
    </div>
  )
}
