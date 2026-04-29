import React, { useEffect, useRef, useState } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

const features = [
  {
    id: 1,
    title: "Omniscient",
    subtitle: "Detection",
    description:
      "Quantum algorithms analyze 47 billion data points per second, identifying threats invisible to human perception.",
    image:
      "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=400&auto=format&fit=crop",
  },
  {
    id: 2,
    title: "Predictive",
    subtitle: "Intelligence",
    description:
      "Machine learning models predict criminal behavior patterns up to 72 hours in advance with unprecedented precision.",
    image:
      "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=400&auto=format&fit=crop",
  },
  {
    id: 3,
    title: "Global",
    subtitle: "Nexus",
    description:
      "Seamlessly integrated network spanning 195 countries, 50,000 satellites, and 2 million ground stations.",
    image:
      "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=400&auto=format&fit=crop",
  },
  {
    id: 4,
    title: "Quantum",
    subtitle: "Processing",
    description:
      "Advanced quantum computing cores process complex threat matrices in real-time across continental networks.",
    image:
      "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?q=80&w=400&auto=format&fit=crop",
  },
];

const Features = () => {
  const containerRef = useRef(null);
  const titleRef = useRef(null);
  const cardsContainerRef = useRef(null);
  const cardRefs = useRef([]);
  const [activeIndex, setActiveIndex] = useState(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      const title = titleRef.current;
      const cards = cardRefs.current;
      const cardsContainer = cardsContainerRef.current;

      const mm = gsap.matchMedia();

      mm.add("(min-width: 768px)", () => {
        // Desktop Animation - Complex Pinning & 3D Scroll
        const title = titleRef.current;
        const cards = cardRefs.current;
        const cardsContainer = cardsContainerRef.current;

        gsap.set(title, {
          rotateX: 80,
          y: 100,
          z: -200,
          opacity: 0,
          transformOrigin: "bottom center",
          force3D: true,
        });

        gsap.set(cards, {
          rotateX: 90,
          y: 150,
          z: -100,
          opacity: 0,
          scale: 0.9,
          transformOrigin: "center bottom",
          force3D: true,
        });

        const tl = gsap.timeline({
          scrollTrigger: {
            trigger: containerRef.current,
            start: "top top",
            end: "+=300%",
            scrub: 1.5,
            pin: true,
            pinSpacing: true,
            anticipatePin: 1,
          },
        });

        tl.to(title, {
          rotateX: 0,
          y: 0,
          z: 0,
          opacity: 1,
          duration: 2,
          ease: "power3.out",
        })
          .to(
            title,
            {
              scale: 0.8,
              y: -200,
              transformOrigin: "top center",
              duration: 1.5,
              ease: "power2.inOut",
            },
            "+=0.2",
          )
          .to(
            cards,
            {
              rotateX: 0,
              y: 0,
              z: 0,
              opacity: 1,
              scale: 1,
              stagger: {
                amount: 0.8,
                from: "center",
              },
              duration: 3,
              ease: "power4.out",
            },
            "<0.5",
          )
          .to({}, { duration: 2 })
          .to([title, cardsContainer], {
            y: -300,
            z: 200,
            opacity: 0,
            duration: 2,
            ease: "power2.in",
          });
      });

      mm.add("(max-width: 767px)", () => {
        // Mobile Animation - Simple Fade In
        const title = titleRef.current;
        const cards = cardRefs.current;

        // Ensure elements are visible/reset from any desktop states
        gsap.set([title, ...cards], { clearProps: "all" });
        gsap.set(containerRef.current, { height: "auto" });

        // Animate Title
        gsap.from(title, {
          scrollTrigger: {
            trigger: title,
            start: "top 80%",
          },
          y: 30,
          opacity: 0,
          duration: 1,
          ease: "power2.out",
        });

        // Animate Cards Staggered
        gsap.from(cards, {
          scrollTrigger: {
            trigger: cardsContainerRef.current,
            start: "top 75%",
          },
          y: 50,
          opacity: 0,
          duration: 0.8,
          stagger: 0.2,
          ease: "power2.out",
        });
      });

      ScrollTrigger.refresh();
    }, containerRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      id="features"
      ref={containerRef}
      className="relative w-full overflow-hidden"
      style={{
        // On mobile, let content flow naturally. On desktop, GSAP handles pinning/height.
        height: "auto",
        perspective: "1000px",
      }}
    >
      {/* Wrapper */}
      <div className="min-h-screen flex flex-col items-center justify-center py-20 md:py-0 md:h-screen md:sticky md:top-0">
        <div className="absolute inset-0 bg-grid-pattern pointer-events-none" />

        <div
          className="relative w-full h-full flex flex-col items-center justify-center z-10"
          style={{ transformStyle: "preserve-3d" }}
        >
          {/* Title */}
          <div
            ref={titleRef}
            className="relative mt-20 md:mt-0 md:absolute md:top-[35%] text-center z-20"
            style={{ willChange: "transform, opacity" }}
          >
            <h2 className="text-5xl md:text-9xl font-black text-neutral-900 tracking-tighter uppercase leading-none">
              Features
            </h2>
            <p className="mt-2 text-sm font-mono text-neutral-500 tracking-[0.5em] uppercase">
              System Capabilities
            </p>
          </div>

          {/* Cards */}
          <div
            ref={cardsContainerRef}
            className="relative top-0 mt-12 md:mt-0 md:top-[15%] w-full max-w-7xl px-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 pb-20 md:pb-0"
            style={{ perspective: "1000px" }}
          >
            {features.map((feature, index) => (
              <div
                key={feature.id}
                ref={(el) => (cardRefs.current[index] = el)}
                className="relative h-auto md:h-100 w-full"
                style={{ willChange: "transform, opacity" }}
                onMouseEnter={() =>
                  window.innerWidth >= 768 && setActiveIndex(index)
                }
                onMouseLeave={() =>
                  window.innerWidth >= 768 && setActiveIndex(null)
                }
                onClick={() =>
                  window.innerWidth < 768 &&
                  setActiveIndex(activeIndex === index ? null : index)
                }
              >
                {/* 
                     Mobile Optimization:
                     - Instead of absolute positioning expansion which breaks flow, we use a more standard card layout on mobile.
                     - Or we simply allow the expansion logic but ensure z-index handles it well. 
                     - Current logic: 'h-[500px] -top-12 z-50'
                     - On mobile, 'relative' positioning is safer for content flow.
                */}
                <div
                  className={`
                    relative md:absolute inset-0 w-full bg-white border border-neutral-200/60 shadow-lg overflow-hidden transition-all duration-500 ease-[cubic-bezier(0.23,1,0.32,1)]
                    ${activeIndex === index ? "md:h-125 md:-top-12 z-50 shadow-2xl scale-100 md:scale-105" : "h-full z-10 scale-100"}
                    ${activeIndex !== null && activeIndex !== index && window.innerWidth >= 768 ? "opacity-40 blur-[2px] scale-95" : "opacity-100"}
                  `}
                >
                  {/* Image Section */}
                  <div
                    className={`
                      w-full bg-neutral-900 relative overflow-hidden transition-all duration-500
                      ${activeIndex === index ? "h-64 md:h-52 opacity-100" : "h-48 md:h-0 opacity-100 md:opacity-0"}
                    `}
                  >
                    <div className="absolute inset-0 bg-neutral-900/20 z-10" />
                    <img
                      src={feature.image}
                      alt={feature.title}
                      loading="lazy"
                      className="w-full h-full object-cover grayscale transition-transform duration-1000 scale-110 hover:scale-100"
                    />
                  </div>

                  {/* Content Section */}
                  <div className="p-6 md:p-8 flex flex-col h-full bg-white">
                    <div
                      className={`h-1 bg-black mb-4 md:mb-6 transition-all duration-500 ${activeIndex === index ? "w-full" : "w-12 text-neutral-300"}`}
                    />
                    <h3 className="text-2xl md:text-3xl font-bold uppercase tracking-tight">
                      {feature.title}
                    </h3>
                    <p className="text-xs font-mono uppercase tracking-widest text-neutral-400 mb-4 md:mb-6">
                      {feature.subtitle}
                    </p>

                    <div
                      className={`transition-all duration-500 ${
                        activeIndex === index
                          ? "opacity-100 max-h-40 translate-y-0"
                          : "opacity-100 md:opacity-0 max-h-40 md:max-h-0 translate-y-0 md:translate-y-8"
                      }`}
                    >
                      <p className="text-neutral-600 text-sm leading-relaxed block">
                        {feature.description}
                      </p>
                    </div>

                    <div className="mt-6 md:mt-auto flex justify-between items-end border-t border-neutral-100 pt-4">
                      <span className="text-[10px] font-mono uppercase tracking-widest text-neutral-500">
                        {activeIndex === index ? "System Active" : "Standby"}
                      </span>
                      <span className="text-2xl md:text-4xl font-black text-neutral-200">
                        0{feature.id}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;
