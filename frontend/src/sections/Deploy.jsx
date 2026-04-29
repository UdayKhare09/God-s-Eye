import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";

const Deploy = () => {
  const navigate = useNavigate();
  function init() {
    navigate("/dashboard");
  }
  return (
    <section
      id="deploy"
      className="min-h-screen w-full py-24 relative flex items-center justify-center"
    >
      <div className="container mx-auto px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-sm font-mono text-neutral-500 mb-6 tracking-widest">
            DEPLOYMENT
          </h2>
          <h3 className="text-6xl md:text-6xl  font-bold text-neutral-900 mb-8">
            Secure Your Sector
          </h3>
          <p className="text-lg text-neutral-600 max-w-2xl mx-auto mb-12 font-light">
            Initiate protocol 89-B to begin local system integration.
            Authorization required for higher tier access.
          </p>
          <button onClick={init} className="px-8 py-4 bg-neutral-900 text-white font-mono text-sm tracking-widest hover:bg-neutral-800 transition-colors uppercase rounded-full">
            Initialize System
          </button>
        </motion.div>
      </div>
    </section>
  );
};

export default Deploy;
