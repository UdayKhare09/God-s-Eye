import React, { Suspense, lazy, useEffect } from "react";
import Loader from "../components/Loader";
import Lenis from "lenis";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

const Hero = lazy(() => import("../sections/Hero"));
const Showcase = lazy(() => import("../sections/Showcase"));
const Vision = lazy(() => import("../sections/Vision"));
const PhotoCarousel = lazy(() => import("../sections/PhotoCarousel"));
const Features = lazy(() => import("../sections/Features"));
const Deploy = lazy(() => import("../sections/Deploy"));
const Team = lazy(() => import("../sections/Team"));
const Footer = lazy(() => import("../sections/Footer"));

gsap.registerPlugin(ScrollTrigger);

export default function LandingPage() {
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      smoothWheel: true,
    });

    lenis.on("scroll", ScrollTrigger.update);

    function raf(time) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);

    ScrollTrigger.refresh();

    return () => {
      lenis.destroy();
    };
  }, []);

  return (
    <div className="min-h-screen font-medium text-neutral-900 bg-[#F5F2EB]">
      {/* Loader now overlays instead of blocking */}
      <Loader />

      {/* <Navbar /> */}

      <div className="relative z-10 bg-grid-pattern mb-[100vh]">
        <Suspense fallback={null}>
          <Hero />
          <Showcase />
          <Vision />
        </Suspense>
      </div>

      <div className="fixed inset-0 z-0 bg-[#F5F2EB]">
        <Suspense fallback={null}>
          <PhotoCarousel />
        </Suspense>
      </div>

      <div className="relative z-10 bg-grid-pattern">
        <Suspense fallback={null}>
          <Features />
          <Deploy />
          <Team />
          <Footer />
        </Suspense>
      </div>
    </div>
  );
}
