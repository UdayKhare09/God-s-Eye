import React, { useRef } from "react";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { Home, Eye, Layers, Zap, Users } from "lucide-react";

const navItems = [
  { name: "Home", icon: Home, id: "home" },
  { name: "Protocol", icon: Eye, id: "vision" },
  { name: "Features", icon: Layers, id: "features" },
  { name: "Deploy", icon: Zap, id: "deploy" },
  { name: "Team", icon: Users, id: "team" },
];

function DockItem({ item, mousePos }) {
  const ref = useRef(null);

  // Calculate distance from mouse to the center of this item
  const distance = useTransform(mousePos, (val) => {
    const bounds = ref.current?.getBoundingClientRect() ?? { x: 0, width: 0 };
    return val - bounds.x - bounds.width / 2;
  });

  // Map distance to width/height/scale
  const sizeSync = useTransform(distance, [-150, 0, 150], [40, 55, 40]);
  const size = useSpring(sizeSync, { mass: 0.1, stiffness: 150, damping: 12 });

  return (
    <motion.div
      ref={ref}
      style={{ width: size, height: size }}
      className="bg-black/80 backdrop-blur-md rounded-full flex items-center justify-center cursor-pointer border border-white/10 relative group"
      onClick={() => {
        const element = document.getElementById(item.id);
        if (element) {
          element.scrollIntoView({ behavior: "smooth" });
        }
        // Reset scale on click (especially for mobile tap)
        mousePos.set(Infinity);
      }}
    >
      <item.icon className="text-white w-1/2 h-1/2" strokeWidth={1.5} />

      {/* Tooltip */}
      <div className="absolute top-full mt-2 px-2 py-1 bg-black text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">
        {item.name}
      </div>
    </motion.div>
  );
}

const Navbar = () => {
  const mousePos = useMotionValue(Infinity);

  return (
    <div
      className="fixed top-4 left-0 w-full z-50 pointer-events-none flex justify-center items-start"
      onMouseMove={(e) => mousePos.set(e.clientX)}
      onMouseLeave={() => mousePos.set(Infinity)}
    >
      <div className="bg-white/10 backdrop-blur-sm border border-black/5 p-2 rounded-full flex flex-row gap-4 pointer-events-auto shadow-xl">
        {navItems.map((item) => (
          <DockItem key={item.id} item={item} mousePos={mousePos} />
        ))}
      </div>
    </div>
  );
};

export default Navbar;
