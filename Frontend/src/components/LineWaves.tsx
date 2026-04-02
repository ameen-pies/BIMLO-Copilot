import { Renderer, Program, Mesh, Triangle } from 'ogl';
import { useEffect, useRef, useState } from 'react';

/**
 * LineWaves Component
 * 
 * A high-performance WebGL background component that adapts to light and dark modes.
 * It uses the OGL library for efficient rendering and supports interactive mouse effects.
 * Optimized for clarity in light mode, using a fresh Sky Blue palette and refined alpha blending.
 */

function hexToVec3(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  if (h.length === 3) {
    return [
      parseInt(h[0] + h[0], 16) / 255,
      parseInt(h[1] + h[1], 16) / 255,
      parseInt(h[2] + h[2], 16) / 255
    ];
  }
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255
  ];
}

const vertexShader = `
attribute vec2 uv;
attribute vec2 position;
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position, 0, 1);
}
`;

const fragmentShader = `
precision highp float;

uniform float uTime;
uniform vec3 uResolution;
uniform float uSpeed;
uniform float uInnerLines;
uniform float uOuterLines;
uniform float uWarpIntensity;
uniform float uRotation;
uniform float uEdgeFadeWidth;
uniform float uColorCycleSpeed;
uniform float uBrightness;
uniform vec3 uColor1;
uniform vec3 uColor2;
uniform vec3 uColor3;
uniform vec2 uMouse;
uniform float uMouseInfluence;
uniform bool uEnableMouse;
uniform bool uIsDark;

#define HALF_PI 1.5707963

float hashF(float n) {
  return fract(sin(n * 127.1) * 43758.5453123);
}

float smoothNoise(float x) {
  float i = floor(x);
  float f = fract(x);
  float u = f * f * (3.0 - 2.0 * f);
  return mix(hashF(i), hashF(i + 1.0), u);
}

float displaceA(float coord, float t) {
  float result = sin(coord * 2.123) * 0.2;
  result += sin(coord * 3.234 + t * 4.345) * 0.1;
  result += sin(coord * 0.589 + t * 0.934) * 0.5;
  return result;
}

float displaceB(float coord, float t) {
  float result = sin(coord * 1.345) * 0.3;
  result += sin(coord * 2.734 + t * 3.345) * 0.2;
  result += sin(coord * 0.189 + t * 0.934) * 0.3;
  return result;
}

vec2 rotate2D(vec2 p, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

void main() {
  vec2 coords = gl_FragCoord.xy / uResolution.xy;
  coords = coords * 2.0 - 1.0;
  coords = rotate2D(coords, uRotation);

  float halfT = uTime * uSpeed * 0.5;
  float fullT = uTime * uSpeed;

  float mouseWarp = 0.0;
  if (uEnableMouse) {
    vec2 mPos = rotate2D(uMouse * 2.0 - 1.0, uRotation);
    float mDist = length(coords - mPos);
    mouseWarp = uMouseInfluence * exp(-mDist * mDist * 4.0);
  }

  float warpAx = coords.x + displaceA(coords.y, halfT) * uWarpIntensity + mouseWarp;
  float warpAy = coords.y - displaceA(coords.x * cos(fullT) * 1.235, halfT) * uWarpIntensity;
  float warpBx = coords.x + displaceB(coords.y, halfT) * uWarpIntensity + mouseWarp;
  float warpBy = coords.y - displaceB(coords.x * sin(fullT) * 1.235, halfT) * uWarpIntensity;

  vec2 fieldA = vec2(warpAx, warpAy);
  vec2 fieldB = vec2(warpBx, warpBy);
  vec2 blended = mix(fieldA, fieldB, mix(fieldA, fieldB, 0.5));

  float fadeTop = smoothstep(uEdgeFadeWidth, uEdgeFadeWidth + 0.4, blended.y);
  float fadeBottom = smoothstep(-uEdgeFadeWidth, -(uEdgeFadeWidth + 0.4), blended.y);
  float vMask = 1.0 - max(fadeTop, fadeBottom);

  float tileCount = mix(uOuterLines, uInnerLines, vMask);
  float scaledY = blended.y * tileCount;
  float nY = smoothNoise(abs(scaledY));

  float ridge = pow(
    step(abs(nY - blended.x) * 2.0, HALF_PI) * cos(2.0 * (nY - blended.x)),
    5.0
  );

  float lines = 0.0;
  for (float i = 1.0; i < 3.0; i += 1.0) {
    lines += pow(max(fract(scaledY), fract(-scaledY)), i * 2.0);
  }

  float pattern = vMask * lines;

  float cycleT = fullT * uColorCycleSpeed;
  float rChannel = (pattern + lines * ridge) * (cos(blended.y + cycleT * 0.234) * 0.5 + 1.0);
  float gChannel = (pattern + vMask * ridge) * (sin(blended.x + cycleT * 1.745) * 0.5 + 1.0);
  float bChannel = (pattern + lines * ridge) * (cos(blended.x + cycleT * 0.534) * 0.5 + 1.0);

  vec3 col = (rChannel * uColor1 + gChannel * uColor2 + bChannel * uColor3) * uBrightness;
  
  // Refined alpha calculation to prevent muddy shadows in light mode
  float alpha;
  if (uIsDark) {
    alpha = clamp(length(col), 0.0, 1.0);
  } else {
    // In light mode, we use a sharper alpha curve to keep the lines crisp and avoid gray halos
    alpha = clamp(length(col) * 1.5, 0.0, 0.8);
  }

  gl_FragColor = vec4(col, alpha);
}
`;

interface LineWavesProps {
  speed?: number;
  innerLineCount?: number;
  outerLineCount?: number;
  warpIntensity?: number;
  rotation?: number;
  edgeFadeWidth?: number;
  colorCycleSpeed?: number;
  brightness?: number;
  lightColors?: [string, string, string];
  darkColors?: [string, string, string];
  enableMouseInteraction?: boolean;
  mouseInfluence?: number;
  className?: string;
  isDark?: boolean;
}

export default function LineWaves({
  speed = 0.3,
  innerLineCount = 32.0,
  outerLineCount = 36.0,
  warpIntensity = 1.0,
  rotation = -45,
  edgeFadeWidth = 0.0,
  colorCycleSpeed = 1.0,
  brightness = 0.2,
  // Clean, vibrant Sky Blue palette for light mode
  lightColors = ['#0ea5e9', '#38bdf8', '#0284c7'], 
  // Neon Blue palette for dark mode
  darkColors = ['#3b82f6', '#60a5fa', '#2563eb'],  
  enableMouseInteraction = true,
  mouseInfluence = 2.0,
  className = "",
  isDark: externalIsDark
}: LineWavesProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const programRef = useRef<Program | null>(null);
  const [internalIsDark, setInternalIsDark] = useState(false);

  useEffect(() => {
    if (externalIsDark !== undefined) {
      setInternalIsDark(externalIsDark);
      return;
    }
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    setInternalIsDark(mediaQuery.matches);
    const handler = (e: MediaQueryListEvent) => setInternalIsDark(e.matches);
    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, [externalIsDark]);

  const activeColors = internalIsDark ? darkColors : lightColors;

  useEffect(() => {
    if (programRef.current) {
      const p = programRef.current;
      p.uniforms.uSpeed.value = speed;
      p.uniforms.uInnerLines.value = innerLineCount;
      p.uniforms.uOuterLines.value = outerLineCount;
      p.uniforms.uWarpIntensity.value = warpIntensity;
      p.uniforms.uRotation.value = (rotation * Math.PI) / 180;
      p.uniforms.uEdgeFadeWidth.value = edgeFadeWidth;
      p.uniforms.uColorCycleSpeed.value = colorCycleSpeed;
      p.uniforms.uBrightness.value = brightness;
      p.uniforms.uColor1.value = hexToVec3(activeColors[0]);
      p.uniforms.uColor2.value = hexToVec3(activeColors[1]);
      p.uniforms.uColor3.value = hexToVec3(activeColors[2]);
      p.uniforms.uMouseInfluence.value = mouseInfluence;
      p.uniforms.uEnableMouse.value = enableMouseInteraction;
      p.uniforms.uIsDark.value = internalIsDark;
    }
  }, [speed, innerLineCount, outerLineCount, warpIntensity, rotation, edgeFadeWidth, colorCycleSpeed, brightness, activeColors, enableMouseInteraction, mouseInfluence, internalIsDark]);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    
    const renderer = new Renderer({ alpha: true, premultipliedAlpha: false });
    const gl = renderer.gl;
    gl.clearColor(0, 0, 0, 0);

    let currentMouse = [0.5, 0.5];
    let targetMouse = [0.5, 0.5];

    function handleMouseMove(e: MouseEvent) {
      const rect = gl.canvas.getBoundingClientRect();
      targetMouse = [
        (e.clientX - rect.left) / rect.width,
        1.0 - (e.clientY - rect.top) / rect.height
      ];
    }

    function handleMouseLeave() {
      targetMouse = [0.5, 0.5];
    }

    function resize() {
      const w = container.offsetWidth || window.innerWidth;
      const h = container.offsetHeight || window.innerHeight;
      renderer.setSize(w, h);
      if (programRef.current) {
        programRef.current.uniforms.uResolution.value = [gl.canvas.width, gl.canvas.height, gl.canvas.width / gl.canvas.height];
      }
    }
    
    window.addEventListener('resize', resize);

    const geometry = new Triangle(gl);
    const program = new Program(gl, {
      vertex: vertexShader,
      fragment: fragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: [0, 0, 0] },
        uSpeed: { value: speed },
        uInnerLines: { value: innerLineCount },
        uOuterLines: { value: outerLineCount },
        uWarpIntensity: { value: warpIntensity },
        uRotation: { value: (rotation * Math.PI) / 180 },
        uEdgeFadeWidth: { value: edgeFadeWidth },
        uColorCycleSpeed: { value: colorCycleSpeed },
        uBrightness: { value: brightness },
        uColor1: { value: hexToVec3(activeColors[0]) },
        uColor2: { value: hexToVec3(activeColors[1]) },
        uColor3: { value: hexToVec3(activeColors[2]) },
        uMouse: { value: new Float32Array([0.5, 0.5]) },
        uMouseInfluence: { value: mouseInfluence },
        uEnableMouse: { value: enableMouseInteraction },
        uIsDark: { value: internalIsDark }
      }
    });

    programRef.current = program;
    resize();

    const mesh = new Mesh(gl, { geometry, program });
    container.appendChild(gl.canvas);

    if (enableMouseInteraction) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseleave', handleMouseLeave);
    }

    let animationFrameId: number;

    function update(time: number) {
      animationFrameId = requestAnimationFrame(update);
      program.uniforms.uTime.value = time * 0.001;

      if (enableMouseInteraction) {
        currentMouse[0] += 0.05 * (targetMouse[0] - currentMouse[0]);
        currentMouse[1] += 0.05 * (targetMouse[1] - currentMouse[1]);
        program.uniforms.uMouse.value[0] = currentMouse[0];
        program.uniforms.uMouse.value[1] = currentMouse[1];
      } else {
        program.uniforms.uMouse.value[0] = 0.5;
        program.uniforms.uMouse.value[1] = 0.5;
      }

      renderer.render({ scene: mesh });
    }
    animationFrameId = requestAnimationFrame(update);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', resize);
      if (enableMouseInteraction) {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseleave', handleMouseLeave);
      }
      if (container.contains(gl.canvas)) {
        container.removeChild(gl.canvas);
      }
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      programRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div 
      ref={containerRef} 
      className={`line-waves-container ${className}`}
      style={{ 
        width: '100%', 
        height: '100%', 
        overflow: 'hidden',
        position: 'absolute',
        top: 0,
        left: 0,
        zIndex: -1,
        pointerEvents: 'none'
      }} 
    />
  );
}
