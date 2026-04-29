import React, { Suspense, lazy } from "react";
import { motion } from "framer-motion";

const CanvasContainer = lazy(() => import("../components/CanvasContainer"));

const Hero = () => {
  function init() {
    window.open(
      "https://github.com/tanishqtiwari7",
      "_blank",
      "noopener,noreferrer",
    );
  }

  return (
    <section
      id="home"
      className="relative h-screen w-full flex items-center overflow-hidden"
    >
      <Suspense fallback={null}>
        <CanvasContainer />
      </Suspense>

      <div className="container mx-auto px-6 md:ml-10 relative z-10 grid grid-cols-1 items-center h-full pointer-events-none">
        <div className="max-w-3xl mt-50 md:mt-12 text-center md:text-left">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, ease: "easeOut" }}
          >
            <h1 className="text-8xl sm:text-7xl md:text-9xl font-bold text-neutral-900 tracking-tighter pb-4">
              GOD'S EYE
            </h1>
          </motion.div>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 1 }}
            className="text-neutral-600 text-lg sm:text-2xl md:text-2xl mt-4 md:mt-6 font-medium leading-relaxed max-w-xl mx-auto md:mx-0"
          >
            Omnipresent surveillance architecture.
            <br />
            See everything. Miss nothing.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1, duration: 0.5 }}
            className="mt-8 md:mt-12 pointer-events-auto inline-block"
          >
            <button
              onClick={init}
              className="bg-neutral-900 text-white px-8 py-4 md:px-10 md:py-5 rounded-full font-bold text-sm md:text-base transition-all duration-300 hover:bg-neutral-700 hover:scale-105 shadow-xl hover:shadow-2xl"
            >
              INITIALIZE SYSTEM
            </button>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
