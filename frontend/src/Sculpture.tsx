import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { usePreferences } from "./state";
export default function Sculpture({
  shape,
  onShape,
}: {
  shape: number;
  onShape: (x: number) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const shapeRef = useRef(shape);
  const [failed, setFailed] = useState(false);
  const { motion } = usePreferences();
  useEffect(() => {
    shapeRef.current = shape;
  }, [shape]);
  useEffect(() => {
    if (!host.current) return;
    const container = host.current;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true,
        powerPreference: "low-power",
      });
    } catch {
      setFailed(true);
      return;
    }
    let disposed = false,
      frame = 0,
      visible = true,
      dragging = false,
      lastX = 0,
      spin = 0,
      targetX = 0,
      lastTime = 0,
      slow = 0,
      lastShape = -1;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setClearColor(0, 0);
    container.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("aria-hidden", "true");
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
    camera.position.set(0, 0.2, 6.5);
    const group = new THREE.Group();
    scene.add(group);
    group.rotation.set(0.2, -0.18, -0.2);
    const geometries: THREE.BufferGeometry[] = [];
    const materials = [
      new THREE.MeshStandardMaterial({
        color: 0x222724,
        roughness: 0.52,
        metalness: 0.22,
        side: THREE.DoubleSide,
      }),
      new THREE.MeshStandardMaterial({
        color: 0x263bea,
        roughness: 0.36,
        metalness: 0.2,
        side: THREE.DoubleSide,
      }),
    ];
    const bands = 100,
      steps = 74;
    for (let band = 0; band < bands; band++) {
      const geo = new THREE.BufferGeometry();
      const points = new Float32Array((steps + 1) * 2 * 3);
      const indices = [];
      for (let j = 0; j < steps; j++) {
        const a = j * 2;
        indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
      geo.setAttribute("position", new THREE.BufferAttribute(points, 3));
      geo.setIndex(indices);
      geometries.push(geo);
      group.add(new THREE.Mesh(geo, materials[band > 47 && band < 54 ? 1 : 0]));
    }
    scene.add(new THREE.AmbientLight(0xffffff, 2.6));
    const key = new THREE.DirectionalLight(0xffffff, 4);
    key.position.set(-2, 4, 4);
    scene.add(key);
    const resize = () => {
      const { width, height } = container.getBoundingClientRect();
      if (!width || !height) return;
      renderer.setSize(width, height);
      group.scale.setScalar(window.innerWidth > 900 ? 0.72 : 1);
      camera.aspect = width / height;
      camera.position.z = width < 550 ? 7.7 : 6.5;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    const onLost = (event: Event) => {
      event.preventDefault();
      disposed = true;
      cancelAnimationFrame(frame);
      setFailed(true);
    };
    renderer.domElement.addEventListener("webglcontextlost", onLost);
    function render(time: number) {
      if (disposed) return;
      frame = requestAnimationFrame(render);
      if (!visible || document.hidden) return;
      if (!motion && lastShape === shapeRef.current) return;
      lastShape = shapeRef.current;
      if (time - lastTime < 32) return;
      if (lastTime && time - lastTime > 65 && ++slow > 20) {
        renderer.setPixelRatio(1);
        slow = 0;
      }
      lastTime = time;
      const phase = motion ? time * 0.00017 : 0;
      spin += (targetX - spin) * 0.04;
      group.rotation.y = -0.18 + (motion ? spin + Math.sin(phase) * 0.16 : 0);
      const morph = shapeRef.current;
      geometries.forEach((geo, band) => {
        const pos = geo.attributes.position as THREE.BufferAttribute;
        for (let j = 0; j <= steps; j++) {
          const u = (j / steps) * Math.PI * 2;
          for (let side = 0; side < 2; side++) {
            const v = (band / bands + (side * 0.66) / bands) * 2 - 1;
            const angle = u + 0.45 * morph * v;
            const radial = 1.03 + v * 0.38;
            const x = radial * Math.cos(angle) * 1.7;
            const y =
              Math.sin(u * 1.5 + morph * 0.7) * v * 1.48 +
              Math.sin(u * 2 + phase) * 0.14 * morph;
            const z =
              radial * Math.sin(angle) * 0.68 + Math.cos(u * 1.5) * v * 0.9;
            pos.setXYZ(j * 2 + side, x, y, z);
          }
        }
        pos.needsUpdate = true;
        geo.computeVertexNormals();
      });
      renderer.render(scene, camera);
    }
    const intersect = new IntersectionObserver(([e]) => {
      visible = e.isIntersecting;
    });
    intersect.observe(container);
    frame = requestAnimationFrame(render);
    const down = (e: PointerEvent) => {
      dragging = true;
      lastX = e.clientX;
      container.setPointerCapture(e.pointerId);
    };
    const move = (e: PointerEvent) => {
      targetX = (e.offsetX / container.clientWidth - 0.5) * 0.5;
      if (dragging) {
        shapeRef.current = Math.max(
          0,
          Math.min(2, shapeRef.current + (e.clientX - lastX) / 160),
        );
        onShape(shapeRef.current);
        lastX = e.clientX;
      }
    };
    const up = () => {
      dragging = false;
    };
    container.addEventListener("pointerdown", down);
    container.addEventListener("pointermove", move);
    container.addEventListener("pointerup", up);
    container.addEventListener("pointercancel", up);
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      intersect.disconnect();
      geometries.forEach((g) => g.dispose());
      materials.forEach((m) => m.dispose());
      renderer.dispose();
      renderer.domElement.remove();
      container.removeEventListener("pointerdown", down);
      container.removeEventListener("pointermove", move);
      container.removeEventListener("pointerup", up);
      container.removeEventListener("pointercancel", up);
    };
  }, [motion]);
  return (
    <div
      className="sculpture"
      ref={host}
      role="img"
      aria-label="A ribbon sculpture that changes shape; use the slider below to reshape it"
    >
      {failed && (
        <div className="static-sculpture" aria-hidden="true">
          {Array.from({ length: 22 }, (_, i) => (
            <i
              key={i}
              style={{
                transform: `rotate(${i * 7}deg) scale(${0.6 + i * 0.021})`,
              }}
            />
          ))}
        </div>
      )}
    </div>
  );
}
