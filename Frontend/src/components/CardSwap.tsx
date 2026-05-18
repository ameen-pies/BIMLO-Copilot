import React, { Children, cloneElement, forwardRef, isValidElement, useEffect, useMemo, useRef } from 'react';
import './CardSwap.css';

export const Card = forwardRef(({ customClass, ...rest }: any, ref: any) => (
  <div ref={ref} {...rest} className={`card ${customClass ?? ''} ${rest.className ?? ''}`.trim()} />
));
Card.displayName = 'Card';

const makeSlot = (i, distX, distY, total) => ({
  x: i * distX,
  y: -i * distY,
  z: -i * distX * 1.5,
  zIndex: total - i
});

const placeNow = (el, slot, skew) => {
  // Use inline styles instead of gsap.set for initial placement
  const transform = `translate(-50%, -50%) translateX(${slot.x}px) translateY(${slot.y}px) translateZ(${slot.z}px) skewY(${skew}deg)`;
  el.style.transform = transform;
  el.style.zIndex = slot.zIndex;
};

const CardSwap = ({
  width = 500,
  height = 400,
  cardDistance = 60,
  verticalDistance = 70,
  delay = 5000,
  pauseOnHover = false,
  onCardClick,
  skewAmount = 6,
  easing = 'elastic',
  children
}: any) => {
  const config =
    easing === 'elastic'
      ? { ease: 'elastic.out(0.6,0.9)', durDrop: 2, durMove: 2, durReturn: 2, promoteOverlap: 0.9, returnDelay: 0.05 }
      : { ease: 'power1.inOut', durDrop: 0.8, durMove: 0.8, durReturn: 0.8, promoteOverlap: 0.45, returnDelay: 0.2 };

  const childArr = useMemo(() => Children.toArray(children), [children]);
  const refs = useMemo(() => childArr.map(() => React.createRef<HTMLDivElement>()), [childArr.length]);
  const order = useRef(Array.from({ length: childArr.length }, (_, i) => i));
  const tlRef = useRef<any>(null);
  const intervalRef = useRef<any>();
  const container = useRef<HTMLDivElement>(null);
  const gsapRef = useRef<any>(null);

  useEffect(() => {
    let cancelled = false;

    const initGsap = async () => {
      const gsap = (await import('gsap')).gsap ?? (await import('gsap')).default;
      if (cancelled) return;
      gsapRef.current = gsap;

      const total = refs.length;
      refs.forEach((r, i) => {
        if (r.current) {
          const slot = makeSlot(i, cardDistance, verticalDistance, total);
          gsap.set(r.current, {
            x: slot.x, y: slot.y, z: slot.z,
            xPercent: -50, yPercent: -50,
            skewY: skewAmount,
            transformOrigin: 'center center',
            zIndex: slot.zIndex,
            force3D: true
          });
        }
      });

      const swap = () => {
        if (order.current.length < 2) return;
        const [front, ...rest] = order.current;
        const elFront = refs[front].current;
        const tl = gsap.timeline();
        tlRef.current = tl;

        tl.to(elFront, { y: '+=500', duration: config.durDrop, ease: config.ease });
        tl.addLabel('promote', `-=${config.durDrop * config.promoteOverlap}`);
        rest.forEach((idx, i) => {
          const el = refs[idx].current;
          const slot = makeSlot(i, cardDistance, verticalDistance, refs.length);
          tl.set(el, { zIndex: slot.zIndex }, 'promote');
          tl.to(el, { x: slot.x, y: slot.y, z: slot.z, duration: config.durMove, ease: config.ease }, `promote+=${i * 0.15}`);
        });

        const backSlot = makeSlot(refs.length - 1, cardDistance, verticalDistance, refs.length);
        tl.addLabel('return', `promote+=${config.durMove * config.returnDelay}`);
        tl.call(() => { gsap.set(elFront, { zIndex: backSlot.zIndex }); }, undefined, 'return');
        tl.to(elFront, { x: backSlot.x, y: backSlot.y, z: backSlot.z, duration: config.durReturn, ease: config.ease }, 'return');
        tl.call(() => { order.current = [...rest, front]; });
      };

      swap();
      intervalRef.current = window.setInterval(swap, delay);

      if (pauseOnHover) {
        const node = container.current;
        const pause = () => { tlRef.current?.pause(); clearInterval(intervalRef.current); };
        const resume = () => { tlRef.current?.play(); intervalRef.current = window.setInterval(swap, delay); };
        node?.addEventListener('mouseenter', pause);
        node?.addEventListener('mouseleave', resume);
        return () => {
          node?.removeEventListener('mouseenter', pause);
          node?.removeEventListener('mouseleave', resume);
          clearInterval(intervalRef.current);
        };
      }
      return () => clearInterval(intervalRef.current);
    };

    initGsap();
    return () => { cancelled = true; clearInterval(intervalRef.current); };
  }, [cardDistance, verticalDistance, delay, pauseOnHover, skewAmount, easing]);

  const rendered = childArr.map((child, i) =>
    isValidElement(child)
      ? cloneElement(child as any, {
          key: i,
          ref: refs[i],
          style: { width, height, ...(child.props.style ?? {}) },
          onClick: (e: any) => {
            (child.props as any).onClick?.(e);
            onCardClick?.(i);
          }
        })
      : child
  );

  return (
    <div ref={container} className="card-swap-container" style={{ width, height }}>
      {rendered}
    </div>
  );
};

export default CardSwap;
