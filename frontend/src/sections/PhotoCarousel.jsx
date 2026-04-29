import React, { memo } from "react";
import { useScroll, useTransform, motion } from "framer-motion";

const images = [
  // Column 1
  "https://images.unsplash.com/photo-1496559249665-c7e2874707ea?q=80&w=1965&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=2070&auto=format&fit=crop",

  // Column 2
  "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop",

  "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?q=80&w=2070&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1549605659-32d82da3a059?q=80&w=2000&auto=format&fit=crop",
  // Column 3
  "https://images.unsplash.com/photo-1504384308090-c54be3855092?q=80&w=1974&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?q=80&w=2070&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1555255707-c07966088b7b?q=80&w=1932&auto=format&fit=crop",

  // Column 4
  "https://images.unsplash.com/photo-1516110833967-0b5716ca1387?q=80&w=1974&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?q=80&w=2070&auto=format&fit=crop",
  "https://images.unsplash.com/photo-1555255707-c07966088b7b?q=80&w=1932&auto=format&fit=crop",
];

const Column = memo(({ images, y }) => {
  return (
    <motion.div
      style={{ y, willChange: "transform" }}
      className="flex-1 min-w-0 h-[150%] flex flex-col gap-6 relative -top-[25%]"
    >
      {images.map((src, i) => (
        <div
          key={i}
          className="w-full aspect-[2/3] md:aspect-[3/4] rounded-lg overflow-hidden relative shadow-lg"
        >
          <img
            src={src}
            alt="surveillance feed"
            loading="lazy"
            className="object-cover w-full h-full grayscale hover:grayscale-0 transition-all duration-700 ease-in-out hover:scale-110"
          />
        </div>
      ))}
    </motion.div>
  );
});

Column.displayName = "Column";

const PhotoCarousel = () => {
  // We track the scroll of the entire page
  const { scrollYProgress } = useScroll();

  // Use a smaller transform range for mobile to prevent huge gaps
  const yRange =
    typeof window !== "undefined" && window.innerWidth < 768 ? 200 : 400;

  // Map scroll progress to vertical movement
  // Since the component is fixed, we want continuous movement as page scrolls
  const y1 = useTransform(scrollYProgress, [0, 1], [0, -yRange]);
  const y2 = useTransform(scrollYProgress, [0, 1], [0, yRange]);
  const y3 = useTransform(scrollYProgress, [0, 1], [0, -yRange]);
  const y4 = useTransform(scrollYProgress, [0, 1], [0, yRange]);

  return (
    <section className="h-screen w-full bg-[#f5f2eb] relative overflow-hidden flex justify-center items-center">
      {/* Grid Background Pattern inside */}
      <div className="absolute inset-0 z-0 opacity-50 bg-[linear-gradient(rgba(0,0,0,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,0,0,0.03)_1px,transparent_1px)] bg-[size:40px_40px]" />

      <div className="container mx-auto h-[120%] md:h-[120vh] flex gap-2 md:gap-8 p-4 rotate-2 scale-100 md:scale-100 transform origin-center">
        {/* Mobile: 2 Columns */}
        <div className="flex md:hidden w-full gap-2">
          <Column images={images.slice(0, 6)} y={y1} />
          <Column images={images.slice(6, 12)} y={y2} />
        </div>

        {/* Desktop: 4 Columns */}
        <div className="hidden md:flex w-full gap-8">
          <Column images={images.slice(0, 3)} y={y1} />
          <Column images={images.slice(3, 6)} y={y2} />
          <Column images={images.slice(6, 9)} y={y3} />
          <Column images={images.slice(9, 12)} y={y4} />
        </div>
      </div>

      {/* Overlay Text */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-20 mix-blend-difference">
        <h2 className="text-[15vw] font-bold text-white tracking-tighter opacity-100 leading-none">
          SEARCH
        </h2>
      </div>
    </section>
  );
};

export default PhotoCarousel;
