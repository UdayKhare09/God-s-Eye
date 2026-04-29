import React, { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { useProgress } from "@react-three/drei";

const MIN_DURATION = 3000;
const EXIT_DELAY = 400;
const START_DELAY = 400;

const Loader = () => {
  const { active } = useProgress();

  const [displayProgress, setDisplayProgress] = useState(0);
  const [sceneReady, setSceneReady] = useState(false);
  const [visible, setVisible] = useState(true);

  const startTime = useRef(Date.now());
  const finishedRef = useRef(false);

  useEffect(() => {
    const ready = () => setSceneReady(true);
    window.addEventListener("scene-ready", ready);
    return () => window.removeEventListener("scene-ready", ready);
  }, []);

  useEffect(() => {
    let raf;

    const update = () => {
      const elapsed = Date.now() - startTime.current;
      const effectiveElapsed = Math.max(0, elapsed - START_DELAY);
      let baseProgress = Math.min(effectiveElapsed / MIN_DURATION, 1) * 95;

      const canFinish = sceneReady && !active && elapsed >= MIN_DURATION;

      if (canFinish && !finishedRef.current) {
        finishedRef.current = true;
        setDisplayProgress(100);
        setTimeout(() => setVisible(false), EXIT_DELAY);
        return;
      }

      setDisplayProgress(prev => {
        const diff = baseProgress - prev;
        return prev + diff * 0.08;
      });

      raf = requestAnimationFrame(update);
    };

    update();
    return () => cancelAnimationFrame(raf);
  }, [sceneReady, active]);

  if (!visible) return null;

  // Phase 1 (0–60%): outline draws in
  // Phase 2 (60–100%): fill floods in
  const outlineProgress = Math.min(displayProgress / 60, 1);
  const fillOpacity = Math.max(0, (displayProgress - 60) / 40);

  return (
    <motion.div
      className="fixed inset-0 z-[999] flex flex-col items-center justify-center bg-[#F5F2EB] text-black"
      animate={{ opacity: displayProgress >= 100 ? 0 : 1 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      style={{ pointerEvents: "none" }}
    >
      {/* animated grid */}
      <motion.div
        className="absolute inset-0 opacity-5"
        animate={{ opacity: [0.03, 0.08, 0.03] }}
        transition={{ duration: 3, repeat: Infinity }}
        style={{
          backgroundImage: "radial-gradient(#000 1px, transparent 1px)",
          backgroundSize: "22px 22px",
        }}
      />

      {/* SVG */}
      <motion.svg
        viewBox="0 0 512 512"
        className="w-28 h-28 md:w-40 md:h-40 mb-10 relative z-10"
      >
        {/*
          The main body shape + eye cutouts in a single path using evenodd fill rule.
          The eye sub-paths (M ... Z sections after the first Z) create hollow punches.
          Replace the eye ellipse approximations below with your actual eye coordinates.
        */}
        <motion.path
          fillRule="evenodd"
          // Main face path  +  left eye hole  +  right eye hole
          // Eyes are approximated as small rounded diamonds — replace with your real eye paths
          d={`
            M256 32.796c-29.75 0-68.891 11.577-99.588 28.844
            -15.348 8.633-28.61 18.674-37.685 28.884
            C109.65 100.735 105 110.796 105 119.796
            c0 1 .75 3.54 3.62 7.164 2.867 3.623 7.55 8.046 13.503 12.809
            11.906 9.524 28.805 20.496 46.87 32.539
            31.444 20.963 66.69 44.82 87.007 74.048
            20.317-29.229 55.563-53.085 87.008-74.048
            18.064-12.043 34.963-23.015 46.869-32.54
            5.953-4.762 10.636-9.185 13.504-12.808
            2.868-3.623 3.619-6.164 3.619-7.164
            0-9-4.65-19.06-13.727-29.272
            -9.076-10.21-22.337-20.251-37.685-28.884
            C324.89 44.373 285.75 32.796 256 32.796z

            M208 110 L220 98 L232 110 L220 122 Z

            M280 110 L292 98 L304 110 L292 122 Z
          `}
          // Phase 1: stroke draws the outline
          stroke="black"
          strokeWidth="3"
          strokeLinejoin="round"
          // Phase 2: fill floods in (eyes stay hollow via evenodd)
          fill="black"
          fillOpacity={fillOpacity}
          animate={{ pathLength: outlineProgress }}
          transition={{ duration: 0.15, ease: "linear" }}
        />
      </motion.svg>

      <div className="text-sm md:text-base font-mono tracking-[0.45em] uppercase relative z-10 text-neutral-700">
        Iinialising God's Eye
      </div>

      <div className="relative mt-6 w-52 md:w-72 h-[2px] bg-neutral-300 overflow-hidden">
        <motion.div
          className="h-full bg-black"
          style={{ width: `${displayProgress}%` }}
        />
      </div>

      <div className="mt-4 text-xs font-mono text-neutral-500 tracking-widest">
        {Math.floor(displayProgress)}%
      </div>
    </motion.div>
  );
};

export default Loader;