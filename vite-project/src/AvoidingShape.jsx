import { motion, useAnimation, useMotionValue } from "framer-motion";
import { useState, useEffect, useRef } from "react";

const AvoidingShape = () => {
  const controls = useAnimation();
  const shapeRef = useRef(null);
  const timeoutRef = useRef(null);

  const [initialX, setInitialX] = useState(window.innerWidth / 2 - 100);
  const x = useMotionValue(initialX);
  const y = useMotionValue(0);

  useEffect(() => {
    const handleResize = () => {
      setInitialX(window.innerWidth / 2 - 100);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  useEffect(() => {
    x.set(initialX);
  }, [initialX, x]);

  const returnToStartPosition = () => {
    controls.start({
      x: initialX,
      y: 0,
      transition: { type: "spring", stiffness: 100, damping: 30 },
    });
  };

  const handleMouseMove = (e) => {
    if (shapeRef.current) {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      const shapeRect = shapeRef.current.getBoundingClientRect();
      const shapeX = shapeRect.left + shapeRect.width / 2;
      const shapeY = shapeRect.top + shapeRect.height / 2;
      const mouseX = e.clientX;
      const mouseY = e.clientY;

      const distX = shapeX - mouseX;
      const distY = shapeY - mouseY;
      const distance = Math.sqrt(distX * distX + distY * distY);

      if (distance < 150) {
        const newX = x.get() + (distX / distance) * 20;
        const newY = y.get() + (distY / distance) * 20;

        const buffer = 50; // shape size
        const clampedX = Math.max(
          -window.innerWidth / 2 + buffer,
          Math.min(window.innerWidth / 2 - buffer, newX)
        );
        const clampedY = Math.max(
          -window.innerHeight / 2 + buffer,
          Math.min(window.innerHeight / 2 - buffer, newY)
        );

        controls.start({
          x: clampedX,
          y: clampedY,
          transition: { type: "spring", stiffness: 500, damping: 10 },
        });
      } else {
        timeoutRef.current = setTimeout(returnToStartPosition, 200);
      }
    }
  };

  useEffect(() => {
    window.addEventListener("mousemove", handleMouseMove);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [controls, initialX]);

  return (
    <motion.div
      ref={shapeRef}
      className="absolute"
      style={{
        width: 50,
        height: 50,
        background: "#00ff99",
        borderRadius: "10px",
        top: "50%",
        left: "50%",
        x,
        y,
      }}
      animate={controls}
    />
  );
};

export default AvoidingShape;