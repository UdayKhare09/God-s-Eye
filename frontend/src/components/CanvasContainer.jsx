import React, { useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Environment, Float, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import SceneReady from "./SceneReady";

const MechanicalEye = () => {
  const { scene } = useGLTF("/mechanical_eye.glb");
  const eyeRef = useRef();
  const { viewport } = useThree();
  const isMobile = viewport.width < 5; // Simple check for mobile screen width in Three.js units

  // Track animation progress (1 = start, 0 = end)
  const progress = useRef(0);
  // Lock to prevent re-triggering during the 0.9s window
  const isAnimating = useRef(false);

  // We need to keep a ref to isMobile to avoid issues inside useFrame
  // However, viewport IS updated on resize, so using it directly inside useFrame is safer if we get it from state

  const { invalidate } = useThree();

  useFrame((state, delta) => {
    if (!eyeRef.current) return;

    // Check mobile status dynamically inside the frame loop reacting to resize
    const currentIsMobile = state.viewport.width < 5;

    // --- MOUSE TRACKING ---
    // Smoothly interpolate mouse position for rotation/position
    // pointer.x goes from -1 (left) to 1 (right)
    // pointer.y goes from -1 (bottom) to 1 (top)

    // Target rotation based on mouse
    const targetRotX = -state.pointer.y * 0.5; // Look up/down
    const targetRotY = state.pointer.x * 0.5; // Look left/right

    // Target position based on mouse (parallax)
    // Base X is 3.0 on desktop, 0 on mobile.
    // Base Y is 0 on desktop, 1.5 on mobile (move it up so text can be below/centered)

    // Desktop defaults
    let basePosX = 3.0;
    let basePosY = 0;

    if (currentIsMobile) {
      basePosX = 0;
      basePosY = 1.5;
    }

    const targetPosX = basePosX + state.pointer.x * 0.5;
    const targetPosY = basePosY + state.pointer.y * 0.5;

    // ... rest of useFrame logic
    // Smoothly dampen the values
    // Using simple lerp/damp for values manually because THREE.MathUtils.damp returns value, doesn't mutate.
    eyeRef.current.rotation.x = THREE.MathUtils.damp(
      eyeRef.current.rotation.x,
      targetRotX,
      4,
      delta,
    );
    eyeRef.current.rotation.y = THREE.MathUtils.damp(
      eyeRef.current.rotation.y,
      targetRotY,
      4,
      delta,
    );

    eyeRef.current.position.x = THREE.MathUtils.damp(
      eyeRef.current.position.x,
      targetPosX,
      4,
      delta,
    );
    eyeRef.current.position.y = THREE.MathUtils.damp(
      eyeRef.current.position.y,
      targetPosY,
      4,
      delta,
    );

    // Ensure we keep rendering while moving
    invalidate();
  });

  return (
    <group
      ref={eyeRef}
      onPointerEnter={() => {
        // ONLY trigger if not already animating
        if (!isAnimating.current) {
          isAnimating.current = true;
          progress.current = 1;
          invalidate();
        }
      }}
      // Initial position for first render
      position={[isMobile ? 0 : 2.5, isMobile ? 1.5 : 0, 0]}
      dispose={null}
    >
      <Float
        speed={3}
        rotationIntensity={0.5}
        floatIntensity={0.2}
        floatingRange={[-0.1, 0.1]}
      >
        <primitive object={scene} scale={isMobile ? 0.6 : 1.2} />
      </Float>
    </group>
  );
};

const CanvasContainer = () => {
  return (
    <div className="absolute inset-0 z-0 bg-transparent">
      <Canvas
        camera={{ position: [0, 0, 8], fov: 45 }}
        gl={{
          antialias: false,
          alpha: true,
          powerPreference: "high-performance",
        }}
        dpr={[1, 1.5]}
        frameloop="demand"
        performance={{ min: 0.5 }}
      >
        <ambientLight intensity={1} />
        <spotLight
          position={[10, 10, 10]}
          angle={0.15}
          penumbra={1}
          intensity={2}
        />
        <pointLight position={[-10, -10, -10]} intensity={1} />
        <Environment preset="city" />
        <MechanicalEye />
       <SceneReady onReady={() => window.dispatchEvent(new Event("scene-ready"))} />
      </Canvas>
    </div>
  );
};

export default CanvasContainer;
