import React, { useRef, useEffect, useState } from "react";
import { motion, useScroll, useTransform } from "framer-motion";

const StatusBadge = ({ label, status }) => (
  <div className="flex items-center gap-2 px-4 py-2 rounded-full border border-neutral-200 bg-white/50 backdrop-blur-sm shadow-sm">
    <div
      className={`w-2 h-2 rounded-full ${status === "active" ? "bg-green-500 animate-pulse" : "bg-neutral-400"}`}
    />
    <span className="text-xs font-mono font-bold tracking-widest uppercase text-neutral-600">
      {label}
    </span>
  </div>
);

const TerminalLog = () => {
  const [lines, setLines] = useState([
    "> SYSTEM_INIT...",
    "> CONNECTING_TO_SATELLITE_LINK...",
  ]);

  useEffect(() => {
    const logs = [
      "> ESTABLISHING_SECURE_HANDSHAKE...",
      "> BIOMETRIC_DATABASE_SYNCED [100%]",
      "> GLOBAL_THREAT_ANALYSIS: ACTIVE",
      "> PREDICITVE_ALGORITHMS: RUNNING",
      "> TARGET_ACQUISITION_MODE: STANDBY",
      "> WAITING_FOR_INPUT...",
    ];
    let i = 0;
    const interval = setInterval(() => {
      if (i < logs.length) {
        setLines((prev) => [...prev, logs[i]].slice(-8));
        i++;
      }
    }, 800);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full h-full bg-black p-6 font-mono text-xs md:text-sm text-green-500 overflow-hidden flex flex-col justify-end relative">
      <div className="absolute inset-x-0 top-0 h-8 bg-neutral-900 flex items-center px-3 border-b border-neutral-800 z-20">
        <span className="text-neutral-500 text-[10px]">
          TERMINAL_ACCESS_V4.2
        </span>
      </div>
      <div className="space-y-1 mt-8 z-10 relative">
        {lines.map((line, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
          >
            {line}
          </motion.div>
        ))}
        <motion.div
          animate={{ opacity: [0, 1, 0] }}
          transition={{ duration: 0.8, repeat: Infinity }}
          className="h-4 w-2 bg-green-500 inline-block align-middle ml-1"
        />
      </div>

      {/* Scanline effect */}
      <div className="absolute inset-0 z-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_2px,3px_100%] pointer-events-none opacity-20" />
    </div>
  );
};

const Showcase = () => {
  const containerRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  });

  const y = useTransform(scrollYProgress, [0, 1], [50, -50]);
  const rotate = useTransform(scrollYProgress, [0, 1], [2, -2]);

  return (
    <section
      id="mission"
      ref={containerRef}
      className="min-h-screen w-full py-24 relative flex flex-col items-center justify-center bg-transparent overflow-hidden"
    >
      {/* Grid Pattern manually applied to ensure visibility */}

      <div className="container mx-auto px-6 relative z-10">
        <div className="flex flex-col lg:flex-row items-center gap-16 lg:gap-24">
          {/* Left Side: Mission Statement */}
          <div className="lg:w-1/2 space-y-10">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
            >
              <h2 className="text-6xl md:text-8xl font-black tracking-tighter text-neutral-900 mb-8 uppercase leading-[0.9]">
                Total <br />
                <span className="text-neutral-400">Recall</span>
              </h2>

              <div className="pl-6 border-l-2 border-neutral-900">
                <p className="text-xl md:text-2xl text-neutral-800 font-medium max-w-lg leading-relaxed">
                  "In a world of chaos, clarity is power. We don't just watch;
                  we understand. We provide the eyes that never blink, the
                  memory that never fades."
                </p>
              </div>
            </motion.div>

            <div className="flex flex-wrap gap-4">
              <StatusBadge label="System Online" status="active" />
              <StatusBadge label="Encrypted" status="active" />
              <StatusBadge label="Global Reach" status="active" />
            </div>
          </div>

          {/* Right Side: Visual Terminal */}
          <div className="lg:w-1/2 w-full flex justify-center lg:justify-end">
            <motion.div
              style={{ y, rotate }}
              className="relative w-full max-w-lg aspect-square md:aspect-video rounded-xl overflow-hidden shadow-2xl border border-neutral-900/10 bg-white group hover:scale-[1.02] transition-transform duration-500 ease-out"
            >
              <TerminalLog />
            </motion.div>
          </div>
        </div>
      </div>

      {/* Decorative gradient fog - REMOVED for clean grid */}
      <div className="absolute bottom-0 left-0 w-full h-10 bg-gradient-to-t from-[#f5f2eb]/50 to-transparent pointer-events-none z-10" />
    </section>
  );
};
export default Showcase;
