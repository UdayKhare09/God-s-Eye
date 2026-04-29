import React, { useRef, useState } from "react";
import { motion, useScroll, useTransform, AnimatePresence } from "framer-motion";
import { Scan, Database, UserCheck, MapPin, Navigation, ChevronDown } from "lucide-react";

const timelineData = [
  {
    id: "01",
    title: "Target Acquisition",
    subtitle: "Input Phase",
    icon: Database,
    description: "Initiate search by uploading a target photograph. System digitizes the subject's biometric profile for network-wide tracking.",
    details: ["Neural Hash Generation", "Multi-angle Reconstruction", "Legacy Database Cross-referencing"]
  },
  {
    id: "02",
    title: "Omni-Scan",
    subtitle: "Network Processing",
    icon: Scan,
    description: "Simultaneously scanning thousands of active camera feeds across the facility, filtering petabytes of visual data in real-time.",
    details: ["Edge Computing Latency: <15ms", "CCTV Integration protocol: v4.2", "Packet Filtering: 400GB/s"]
  },
  {
    id: "03",
    title: "Facial Recognition",
    subtitle: "AI Verification",
    icon: UserCheck,
    description: "Deep learning neural networks compare captured faces against the localized target profile with 99.9% matching accuracy.",
    details: ["Biometric Landmark Mapping", "Low-light Enhancement AI", "Occlusion Recovery Logic"]
  },
  {
    id: "04",
    title: "Precision Localization",
    subtitle: "Spatial Mapping",
    icon: MapPin,
    description: "Triangulating specific coordinates (Floor 3, Room 304). Mapping target position within the building's 3D architectural model.",
    details: ["LIDAR Mesh Overlay", "Sub-meter Accuracy", "In-building GPS Relay"]
  },
  {
    id: "05",
    title: "Predictive Pathing",
    subtitle: "Live Tracking",
    icon: Navigation,
    description: "Calculating trajectory and movement speed to forecast destination. Displaying a real-time breadcrumb trail of the target's route.",
    details: ["Velocity Vector Analysis", "Heat-map Probability", "Interception Point Calculation"]
  },
];

const isTouchDevice = () =>
  typeof window !== "undefined" &&
  ("ontouchstart" in window || navigator.maxTouchPoints > 0);



/* ---------------- Timeline Item ---------------- */

const TimelineItem = ({ item, index }) => {
  const [expanded, setExpanded] = useState(false);
  const ref = useRef(null);

  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start 85%", "center 60%"],
  });

  const opacity = useTransform(scrollYProgress, [0, 1], [0.25, 1]);
  const y = useTransform(scrollYProgress, [0, 1], [70, 0]);
  const scale = useTransform(scrollYProgress, [0, 1], [0.95, 1]);

  const handleClick = () => {
    if (isTouchDevice()) setExpanded((prev) => !prev);
  };

  return (
    <motion.div
      ref={ref}
      style={{ opacity, y, scale }}
      className={`relative flex flex-col md:flex-row items-start gap-8 md:gap-16 w-full ${
        index % 2 === 0 ? "md:flex-row" : "md:flex-row-reverse"
      }`}
    >
      <div
        className="flex-1 w-full md:w-1/2 group"
        onClick={handleClick}
        onMouseEnter={() => !isTouchDevice() && setExpanded(true)}
        onMouseLeave={() => !isTouchDevice() && setExpanded(false)}
      >
        <div className={`p-6 md:p-8 rounded-2xl border transition-all duration-500 cursor-pointer ${
          expanded
            ? "bg-white border-neutral-300 shadow-[0_20px_60px_rgba(0,0,0,0.12)]"
            : "bg-white border-neutral-900/5 backdrop-blur-md"
        }`}>

          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <span className="text-3xl font-mono font-black text-neutral-300 group-hover:text-neutral-900 transition-colors">
                {item.id}
              </span>

              <div className={`p-2.5 rounded-xl transition-all duration-500 ${
                expanded ? "bg-black text-white" : "bg-neutral-100 text-neutral-400"
              }`}>
                <item.icon size={22} strokeWidth={2}/>
              </div>
            </div>

            <motion.div animate={{ rotate: expanded ? 180 : 0 }}>
              <ChevronDown size={20} className="text-neutral-400"/>
            </motion.div>
          </div>

          <h3 className="text-xl md:text-2xl font-bold text-neutral-800">
            {item.title}
          </h3>

          <p className="text-xs font-mono text-neutral-400 uppercase tracking-widest mb-3">
            {item.subtitle}
          </p>

          <p className="text-neutral-600 leading-relaxed text-sm md:text-base">
            {item.description}
          </p>

          <AnimatePresence>
            {expanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.35 }}
                className="overflow-hidden"
              >
                <div className="pt-6 mt-6 border-t border-neutral-100">
                  <p className="text-xs font-bold text-neutral-900 uppercase mb-3 tracking-tighter">
                    Technical Specs
                  </p>

                  <ul className="space-y-2">
                    {item.details.map((detail, i) => (
                      <li key={i} className="flex items-center gap-2 text-xs text-neutral-500 font-mono">
                        <div className="w-1 h-1 bg-neutral-400 rounded-full"/>
                        {detail}
                      </li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

        </div>
      </div>

      <div className="flex-1 hidden md:block"/>
    </motion.div>
  );
};



/* ---------------- Section ---------------- */

const Vision = () => {
  const containerRef = useRef(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start 60%", "end 20%"], // ensures line finishes at last card
  });

  const scaleY = useTransform(scrollYProgress, [0, 1], [0, 1]);

  return (
    <section
      id="vision"
      ref={containerRef}
      className="pt-32 pb-[40vh] md:pt-40 md:pb-[45vh] relative"
    >
      <div className="container mx-auto px-6">

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          className="text-center mb-32"
        >
          <span className="text-xs font-mono text-neutral-400 tracking-[0.6em] uppercase block mb-4">
            Phase Operations
          </span>

          <h2 className="text-5xl md:text-8xl font-black uppercase tracking-tighter text-neutral-900">
            The Protocol
          </h2>
        </motion.div>

        <div className="relative max-w-5xl mx-auto">

          {/* Scan Line */}
          <div className="absolute left-6 md:left-1/2 md:-translate-x-1/2 top-0 bottom-0 w-[2px] bg-neutral-200">
            <motion.div
              style={{ scaleY, transformOrigin: "top" }}
              className="h-full bg-neutral-900 shadow-[0_0_20px_rgba(0,0,0,0.25)]"
            />
          </div>

          <div className="flex flex-col gap-32 pl-12 md:pl-0">
            {timelineData.map((item, index) => (
              <TimelineItem key={item.id} item={item} index={index} />
            ))}
          </div>

        </div>
      </div>
    </section>
  );
};

export default Vision;