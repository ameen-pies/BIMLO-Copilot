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

/**
 * Map orbital hue (0-360) to an RGB triple.
 * 0 = idle (indigo), 150 = listening (teal), 360 = speaking (blue)
 */
function hueToRgb(h: number): [number, number, number] {
  const s = 0.65;
  const l = 0.55;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

function orbColor(hue: number): [number, number, number] {
  if (hue === 0) return [0.39, 0.40, 0.95];   // indigo
  if (hue === 360) return [0.20, 0.50, 0.95];  // blue
  return hueToRgb(hue);
}

const NetworkBackground = ({
  isDark = true,
  orbHue = 0,
}: {
  isDark?: boolean;
  orbHue?: number;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDarkRef = useRef(isDark);
  const orbHueRef = useRef(orbHue);
  isDarkRef.current = isDark;
  orbHueRef.current = orbHue;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const setCanvasSize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    setCanvasSize();
    window.addEventListener('resize', setCanvasSize);

    const mouse = { x: -1000, y: -1000 };
    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };
    window.addEventListener('mousemove', handleMouseMove);

    // Generate well-spaced nodes
    const nodes: Node[] = [];
    const minDistance = 200;
    const maxAttempts = 30;
    const targetNodes = 12;

    nodes.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
    });

    while (nodes.length < targetNodes) {
      let placed = false;
      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const candidate = {
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
        };
        let tooClose = false;
        for (const node of nodes) {
          const dx = candidate.x - node.x;
          const dy = candidate.y - node.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < minDistance) { tooClose = true; break; }
        }
        if (!tooClose) { nodes.push(candidate); placed = true; break; }
      }
      if (!placed) break;
    }

    // Connect nodes
    const connections: Connection[] = [];
    const usedConnections = new Set<string>();

    nodes.forEach((node, nodeIdx) => {
      const distances = nodes
        .map((otherNode, otherIdx) => {
          if (nodeIdx === otherIdx) return null;
          const dx = otherNode.x - node.x;
          const dy = otherNode.y - node.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          return { idx: otherIdx, distance, node: otherNode };
        })
        .filter(d => d !== null) as Array<{idx: number, distance: number, node: Node}>;

      distances.sort((a, b) => b.distance - a.distance);
      const numConnections = 2 + Math.floor(Math.random() * 2);
      const selectedIndices = new Set<number>();

      for (let i = 0; i < Math.min(numConnections, distances.length); i++) {
        let idx;
        if (i === 0) idx = Math.floor(Math.random() * Math.floor(distances.length / 3));
        else if (i === 1) { const start = Math.floor(distances.length / 3); idx = start + Math.floor(Math.random() * Math.floor(distances.length / 3)); }
        else idx = Math.floor(Math.random() * distances.length);
        while (selectedIndices.has(idx)) idx = Math.floor(Math.random() * distances.length);
        selectedIndices.add(idx);
        const target = distances[idx];
        const key = [nodeIdx, target.idx].sort().join('-');
        if (!usedConnections.has(key)) {
          usedConnections.add(key);
          const dx = target.node.x - node.x;
          const dy = target.node.y - node.y;
          const midY = (node.y + target.node.y) / 2;
          const curvature = target.distance * 0.15;
          connections.push({
            from: node, to: target.node,
            angle: Math.atan2(dy, dx), length: target.distance,
            controlY: midY - curvature,
          });
        }
      }
    });

    const getPointOnCurve = (from: Node, to: Node, controlY: number, t: number) => {
      const controlX = (from.x + to.x) / 2;
      const x = (1 - t) * (1 - t) * from.x + 2 * (1 - t) * t * controlX + t * t * to.x;
      const y = (1 - t) * (1 - t) * from.y + 2 * (1 - t) * t * controlY + t * t * to.y;
      return { x, y };
    };

    // Signals
    const signals: Signal[] = [];
    const createSignal = () => {
      if (connections.length > 0) {
        const conn = connections[Math.floor(Math.random() * connections.length)];
        const initialPos = getPointOnCurve(conn.from, conn.to, conn.controlY, 0);
        signals.push({
          connection: conn, progress: 0, speed: 0.0008 + Math.random() * 0.0012,
          tailLength: 0.18 + Math.random() * 0.12, x: initialPos.x, y: initialPos.y,
          vx: 0, vy: 0, isOrbiting: false, originalProgress: 0, trail: [],
        });
      }
    };
    for (let i = 0; i < 6; i++) createSignal();

    // Current lerped color
    let curColor: [number, number, number] = orbColor(orbHueRef.current);

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const dark = isDarkRef.current;
      const hue  = orbHueRef.current;
      const target = orbColor(hue);

      // Smooth color transition
      curColor = [
        curColor[0] + (target[0] - curColor[0]) * 0.035,
        curColor[1] + (target[1] - curColor[1]) * 0.035,
        curColor[2] + (target[2] - curColor[2]) * 0.035,
      ];

      const [r, g, b] = curColor;
      // Dark mode: lower opacity lines (subtle). Light mode: stronger lines for contrast
      const baseAlpha = dark ? 0.15 : 0.25;
      const nodeAlpha = dark ? 0.25 : 0.40;
      const sigAlpha  = dark ? 0.6  : 0.8;

      // Draw connections
      connections.forEach((conn) => {
        const controlX = (conn.from.x + conn.to.x) / 2;
        ctx.strokeStyle = `rgba(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0},${baseAlpha * 0.6})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(conn.from.x, conn.from.y);
        ctx.quadraticCurveTo(controlX, conn.controlY, conn.to.x, conn.to.y);
        ctx.stroke();

        ctx.strokeStyle = `rgba(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0},${baseAlpha * 0.35})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(conn.from.x, conn.from.y);
        ctx.quadraticCurveTo(controlX, conn.controlY, conn.to.x, conn.to.y);
        ctx.stroke();
      });

      // Draw nodes
      nodes.forEach((node) => {
        ctx.fillStyle = `rgba(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0},${nodeAlpha * 0.6})`;
        ctx.beginPath();
        ctx.arc(node.x, node.y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });

      // Draw signals
      signals.forEach((signal, index) => {
        const conn = signal.connection;
        const mouseInfluence = 180;
        const dx = mouse.x - signal.x;
        const dy = mouse.y - signal.y;
        const distToMouse = Math.sqrt(dx * dx + dy * dy);

        signal.trail.push({x: signal.x, y: signal.y});
        if (signal.trail.length > 60) signal.trail.shift();

        if (distToMouse < mouseInfluence) {
          signal.isOrbiting = true;
          const desiredOrbitRadius = 80;
          const radiusError = distToMouse - desiredOrbitRadius;
          const radialForce = radiusError * 0.015;
          signal.vx += (dx / distToMouse) * radialForce;
          signal.vy += (dy / distToMouse) * radialForce;
          const orbitalForce = 0.5;
          signal.vx += (-dy / distToMouse) * orbitalForce;
          signal.vy += (dx / distToMouse) * orbitalForce;
        } else if (signal.isOrbiting) {
          const targetPos = getPointOnCurve(conn.from, conn.to, conn.controlY, signal.progress);
          const returnDx = targetPos.x - signal.x;
          const returnDy = targetPos.y - signal.y;
          const returnDist = Math.sqrt(returnDx * returnDx + returnDy * returnDy);
          if (returnDist < 5) {
            signal.isOrbiting = false;
            signal.trail = [];
            signal.resumeSpeedProgress = 0;
          } else {
            const springForce = 0.08;
            signal.vx += (returnDx / returnDist) * springForce * returnDist * 0.01;
            signal.vy += (returnDy / returnDist) * springForce * returnDist * 0.01;
          }
        }

        if (signal.isOrbiting || (signal.vx !== 0 || signal.vy !== 0)) {
          if (!signal.isOrbiting && signal.resumeSpeedProgress !== undefined && signal.resumeSpeedProgress < 1) {
            signal.resumeSpeedProgress += 0.016;
            const easeOut = 1 - Math.pow(1 - signal.resumeSpeedProgress, 3);
            const pathSpeed = signal.speed * easeOut;
            signal.progress += pathSpeed;
            const newPos = getPointOnCurve(conn.from, conn.to, conn.controlY, signal.progress);
            const blendFactor = signal.resumeSpeedProgress;
            signal.x += signal.vx * (1 - blendFactor) + (newPos.x - signal.x) * blendFactor;
            signal.y += signal.vy * (1 - blendFactor) + (newPos.y - signal.y) * blendFactor;
            signal.vx *= 0.85; signal.vy *= 0.85;
            if (signal.resumeSpeedProgress >= 1) { signal.vx = 0; signal.vy = 0; }
          } else {
            signal.vx *= 0.92; signal.vy *= 0.92;
            signal.x += signal.vx; signal.y += signal.vy;
          }
        } else {
          signal.progress += signal.speed;
          const pos = getPointOnCurve(conn.from, conn.to, conn.controlY, signal.progress);
          signal.x = pos.x; signal.y = pos.y;
        }

        const fadeStartProgress = 0.85;
        const fadeOutFactor = signal.progress < fadeStartProgress ? 1 : 1 - ((signal.progress - fadeStartProgress) / (1 - fadeStartProgress));

        if (signal.progress >= 1.15) {
          signals.splice(index, 1);
          if (Math.random() > 0.85) createSignal();
          return;
        }

        // Signal trail
        if (signal.trail.length > 1) {
          for (let i = 0; i < signal.trail.length - 1; i++) {
            const point = signal.trail[i];
            const nextPoint = signal.trail[i + 1];
            const opacity = (i / signal.trail.length) * sigAlpha * fadeOutFactor;
            ctx.strokeStyle = `rgba(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0},${opacity})`;
            ctx.lineWidth = 2.5 * (i / signal.trail.length);
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(point.x, point.y);
            ctx.lineTo(nextPoint.x, nextPoint.y);
            ctx.stroke();
          }
        }

        // Signal glow
        const glow = ctx.createRadialGradient(signal.x, signal.y, 0, signal.x, signal.y, 10);
        glow.addColorStop(0, `rgba(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0},${0.9 * fadeOutFactor})`);
        glow.addColorStop(0.5, `rgba(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0},${0.4 * fadeOutFactor})`);
        glow.addColorStop(1, `rgba(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0},0)`);
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(signal.x, signal.y, 10, 0, Math.PI * 2);
        ctx.fill();

        // Bright core
        ctx.fillStyle = `rgba(255, 255, 255, ${0.7 * fadeOutFactor})`;
        ctx.beginPath();
        ctx.arc(signal.x, signal.y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });

      if (Math.random() > 0.99 && signals.length < 8) createSignal();
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
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0, opacity: 0.45 }}
    />
  );
};

export default NetworkBackground;