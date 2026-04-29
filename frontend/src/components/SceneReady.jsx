import { useEffect } from "react";
import { useThree } from "@react-three/fiber";

const SceneReady = ({ onReady }) => {
  const { gl, scene, camera } = useThree();

  useEffect(() => {
    const id = requestAnimationFrame(() => {
      // force shader compilation BEFORE loader disappears
      gl.compile(scene, camera);

      // give GPU one frame to present pixels
      requestAnimationFrame(() => {
        if (onReady) onReady();
      });
    });

    return () => cancelAnimationFrame(id);
  }, [gl, scene, camera, onReady]);

  return null;
};

export default SceneReady;