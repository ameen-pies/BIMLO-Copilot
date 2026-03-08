import { useEffect, useRef } from 'react';

interface Node {
  x: number;
  y: number;
}

interface Connection {
  from: Node;
  to: Node;
  angle: number;
  length: number;
  controlY: number;
}

interface Signal {
  connection: Connection;
  progress: number;
  speed: number;
  tailLength: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  isOrbiting: boolean;
  originalProgress: number;
  trail: Array<{x: number, y: number}>;
  resumeSpeedProgress?: number;
}

const NetworkBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const setCanvasSize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);

    // Track mouse position
    const mouse = { x: -1000, y: -1000 };
    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };
    window.addEventListener('mousemove', handleMouseMove);

    // Generate well-spaced nodes using Poisson disc sampling approach
    const nodes: Node[] = [];
    const minDistance = 200; // Minimum distance between nodes (increased)
    const maxAttempts = 30;
    const targetNodes = 12; // Reduced from 20

    // Add first node randomly
    nodes.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
    });

    // Generate more nodes with minimum spacing
    while (nodes.length < targetNodes) {
      let placed = false;
      
      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const candidate = {
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
        };

        // Check distance to all existing nodes
        let tooClose = false;
        for (const node of nodes) {
          const dx = candidate.x - node.x;
          const dy = candidate.y - node.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          
          if (dist < minDistance) {
            tooClose = true;
            break;
          }
        }

        if (!tooClose) {
          nodes.push(candidate);
          placed = true;
          break;
        }
      }

      if (!placed) break; // Can't place more nodes
    }

    // Connect each node to 3-4 OTHER nodes (like countries connected globally)
    const connections: Connection[] = [];
    const usedConnections = new Set<string>();

    nodes.forEach((node, nodeIdx) => {
      // Calculate distances to all other nodes
      const distances = nodes
        .map((otherNode, otherIdx) => {
          if (nodeIdx === otherIdx) return null;
          
          const dx = otherNode.x - node.x;
          const dy = otherNode.y - node.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          return { idx: otherIdx, distance, node: otherNode };
        })
        .filter(d => d !== null) as Array<{idx: number, distance: number, node: Node}>;

      // Sort by distance
      distances.sort((a, b) => b.distance - a.distance);

      // Connect to 2-3 nodes, preferring varied distances
      const numConnections = 2 + Math.floor(Math.random() * 2); // 2 or 3 (reduced from 3-4)
      const selectedIndices = new Set<number>();

      // Pick some far nodes, some medium distance for variety
      for (let i = 0; i < Math.min(numConnections, distances.length); i++) {
        let idx;
        if (i === 0) {
          // First connection: pick from farthest third
          idx = Math.floor(Math.random() * Math.floor(distances.length / 3));
        } else if (i === 1) {
          // Second: medium distance
          const start = Math.floor(distances.length / 3);
          const range = Math.floor(distances.length / 3);
          idx = start + Math.floor(Math.random() * range);
        } else {
          // Rest: anywhere
          idx = Math.floor(Math.random() * distances.length);
        }

        // Make sure we don't pick the same target twice
        while (selectedIndices.has(idx)) {
          idx = Math.floor(Math.random() * distances.length);
        }
        selectedIndices.add(idx);

        const target = distances[idx];
        
        // Create unique key for connection (avoid duplicates)
        const key = [nodeIdx, target.idx].sort().join('-');
        
        if (!usedConnections.has(key)) {
          usedConnections.add(key);
          
          const dx = target.node.x - node.x;
          const dy = target.node.y - node.y;
          const midY = (node.y + target.node.y) / 2;
          const curvature = target.distance * 0.15;
          
          connections.push({
            from: node,
            to: target.node,
            angle: Math.atan2(dy, dx),
            length: target.distance,
            controlY: midY - curvature,
          });
        }
      }
    });

    // Helper: Get point on quadratic curve
    const getPointOnCurve = (from: Node, to: Node, controlY: number, t: number) => {
      const controlX = (from.x + to.x) / 2;
      const x = (1 - t) * (1 - t) * from.x + 2 * (1 - t) * t * controlX + t * t * to.x;
      const y = (1 - t) * (1 - t) * from.y + 2 * (1 - t) * t * controlY + t * t * to.y;
      return { x, y };
    };

    // Create signals
    const signals: Signal[] = [];
    const createSignal = () => {
      if (connections.length > 0) {
        const conn = connections[Math.floor(Math.random() * connections.length)];
        const initialPos = getPointOnCurve(conn.from, conn.to, conn.controlY, 0);
        signals.push({
          connection: conn,
          progress: 0,
          speed: 0.0008 + Math.random() * 0.0012,
          tailLength: 0.18 + Math.random() * 0.12,
          x: initialPos.x,
          y: initialPos.y,
          vx: 0,
          vy: 0,
          isOrbiting: false,
          originalProgress: 0,
          trail: [],
        });
      }
    };

    // Initial signals - fewer
    for (let i = 0; i < 6; i++) { // Increased from 2
      createSignal();
    }

    // Animation loop
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw curved connections
      connections.forEach((conn) => {
        const controlX = (conn.from.x + conn.to.x) / 2;
        
        // Outer tube - visible but subtle
        ctx.strokeStyle = 'rgba(71, 85, 105, 0.15)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(conn.from.x, conn.from.y);
        ctx.quadraticCurveTo(controlX, conn.controlY, conn.to.x, conn.to.y);
        ctx.stroke();

        // Inner tube
        ctx.strokeStyle = 'rgba(100, 116, 139, 0.08)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(conn.from.x, conn.from.y);
        ctx.quadraticCurveTo(controlX, conn.controlY, conn.to.x, conn.to.y);
        ctx.stroke();
      });

      // Draw nodes
      nodes.forEach((node) => {
        ctx.fillStyle = 'rgba(71, 85, 105, 0.25)';
        ctx.beginPath();
        ctx.arc(node.x, node.y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw signals
      signals.forEach((signal, index) => {
        const conn = signal.connection;
        const mouseInfluence = 180; // Distance at which mouse affects signals (increased)
        const dx = mouse.x - signal.x;
        const dy = mouse.y - signal.y;
        const distToMouse = Math.sqrt(dx * dx + dy * dy);

        // Add current position to trail
        signal.trail.push({x: signal.x, y: signal.y});
        if (signal.trail.length > 60) {
          signal.trail.shift(); // Keep trail at max 60 points for longer trails
        }

        if (distToMouse < mouseInfluence) {
          // Mouse is close - orbit around it at greater radius
          signal.isOrbiting = true;
          
          const desiredOrbitRadius = 80; // Larger orbit radius
          const radiusError = distToMouse - desiredOrbitRadius;
          
          // Adjust distance to maintain orbit
          const radialForce = radiusError * 0.015;
          signal.vx += (dx / distToMouse) * radialForce;
          signal.vy += (dy / distToMouse) * radialForce;
          
          // Add orbital velocity (perpendicular to radius)
          const orbitalForce = 0.5;
          signal.vx += (-dy / distToMouse) * orbitalForce;
          signal.vy += (dx / distToMouse) * orbitalForce;
        } else if (signal.isOrbiting) {
          // Was orbiting but mouse moved away - spring back to path
          const targetPos = getPointOnCurve(conn.from, conn.to, conn.controlY, signal.progress);
          const returnDx = targetPos.x - signal.x;
          const returnDy = targetPos.y - signal.y;
          const returnDist = Math.sqrt(returnDx * returnDx + returnDy * returnDy);
          
          if (returnDist < 5) {
            // Close enough to path - transition to path following
            signal.isOrbiting = false;
            // Don't zero velocity - let it naturally decay
            signal.trail = []; // Clear trail when back on path
            signal.resumeSpeedProgress = 0; // Start speed ramp-up
          } else {
            // Spring back
            const springForce = 0.08;
            signal.vx += (returnDx / returnDist) * springForce * returnDist * 0.01;
            signal.vy += (returnDy / returnDist) * springForce * returnDist * 0.01;
          }
        }

        if (signal.isOrbiting || (signal.vx !== 0 || signal.vy !== 0)) {
          // If transitioning back to path, blend velocities
          if (!signal.isOrbiting && signal.resumeSpeedProgress !== undefined && signal.resumeSpeedProgress < 1) {
            // Calculate path velocity
            signal.resumeSpeedProgress += 0.016;
            const easeOut = 1 - Math.pow(1 - signal.resumeSpeedProgress, 3);
            const pathSpeed = signal.speed * easeOut;
            
            // Update progress along path
            signal.progress += pathSpeed;
            const newPos = getPointOnCurve(conn.from, conn.to, conn.controlY, signal.progress);
            
            // Blend between orbit velocity and path velocity
            const blendFactor = signal.resumeSpeedProgress;
            signal.x += signal.vx * (1 - blendFactor) + (newPos.x - signal.x) * blendFactor;
            signal.y += signal.vy * (1 - blendFactor) + (newPos.y - signal.y) * blendFactor;
            
            // Gradually zero out orbit velocity
            signal.vx *= 0.85;
            signal.vy *= 0.85;
            
            // Clear velocity when blend is complete
            if (signal.resumeSpeedProgress >= 1) {
              signal.vx = 0;
              signal.vy = 0;
            }
          } else {
            // Pure orbit movement
            // Apply damping
            signal.vx *= 0.92;
            signal.vy *= 0.92;
            
            signal.x += signal.vx;
            signal.y += signal.vy;
          }
        } else {
          // Normal path movement
          signal.progress += signal.speed;
          const pos = getPointOnCurve(conn.from, conn.to, conn.controlY, signal.progress);
          signal.x = pos.x;
          signal.y = pos.y;
        }

        // Calculate fade out as signal approaches destination node
        const fadeStartProgress = 0.85;
        const fadeOutFactor = signal.progress < fadeStartProgress 
          ? 1 
          : 1 - ((signal.progress - fadeStartProgress) / (1 - fadeStartProgress));

        // Remove if too far along path
        if (signal.progress >= 1.15) {
          signals.splice(index, 1);
          if (Math.random() > 0.85) { // More frequent spawning
            createSignal();
          }
          return;
        }

        // Draw trail
        if (signal.trail.length > 1) {
          for (let i = 0; i < signal.trail.length - 1; i++) {
            const point = signal.trail[i];
            const nextPoint = signal.trail[i + 1];
            const opacity = (i / signal.trail.length) * 0.6 * fadeOutFactor;

            ctx.strokeStyle = `rgba(96, 165, 250, ${opacity})`;
            ctx.lineWidth = 2.5 * (i / signal.trail.length);
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(point.x, point.y);
            ctx.lineTo(nextPoint.x, nextPoint.y);
            ctx.stroke();
          }
        }

        // Draw glow
        const gradient = ctx.createRadialGradient(signal.x, signal.y, 0, signal.x, signal.y, 10);
        gradient.addColorStop(0, `rgba(147, 197, 253, ${0.9 * fadeOutFactor})`);
        gradient.addColorStop(0.5, `rgba(96, 165, 250, ${0.4 * fadeOutFactor})`);
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(signal.x, signal.y, 10, 0, Math.PI * 2);
        ctx.fill();
        
        // Bright core
        ctx.fillStyle = `rgba(255, 255, 255, ${0.8 * fadeOutFactor})`;
        ctx.beginPath();
        ctx.arc(signal.x, signal.y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });

      // Add new signals - less frequent
      if (Math.random() > 0.99 && signals.length < 8) { // Increased from 4
        createSignal();
      }

      requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', setCanvasSize);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none opacity-40"
      style={{ zIndex: 0 }}
    />
  );
};

export default NetworkBackground;