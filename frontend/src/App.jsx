import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';

const SOCKET_URL = 'http://localhost:5001';

function App() {
  const [screen, setScreen] = useState('menu');
  const [gameState, setGameState] = useState({
    score: 0,
    debug_mode: false,
    active: false,
    target_pos: null,
    target_body_part: null,
    frame_width: 640,
    frame_height: 480
  });
  const [handData, setHandData] = useState(null);
  const [hoverIndex, setHoverIndex] = useState(null);
  const [fistTimer, setFistTimer] = useState(0);
  const [cameraReady, setCameraReady] = useState(false);
  const [videoFrame, setVideoFrame] = useState(null);
  
  const socketRef = useRef(null);
  const FIST_HOLD_FRAMES = 30;

  const buttons = [
    { text: 'Single Player', color: '#329632', mode: 'single' },
    { text: 'Two Players', color: '#646464', mode: 'two', disabled: true },
    { text: 'Crazy Multiplayer', color: '#646464', mode: 'crazy', disabled: true }
  ];

  useEffect(() => {
    // Initialize Socket.IO connection
    socketRef.current = io(SOCKET_URL, {
      transports: ['websocket'],
      upgrade: false
    });

    socketRef.current.on('connect', () => {
      console.log('Connected to server');
      socketRef.current.emit('init_camera');
    });

    socketRef.current.on('camera_ready', (data) => {
      if (data.success) {
        setCameraReady(true);
        setGameState(prev => ({
          ...prev,
          frame_width: data.width,
          frame_height: data.height
        }));
        console.log('Camera ready:', data.width, 'x', data.height);
      } else {
        alert('Cannot access webcam: ' + data.error);
      }
    });

    socketRef.current.on('video_frame', (data) => {
      setVideoFrame('data:image/jpeg;base64,' + data.frame);
    });

    socketRef.current.on('hand_data', (data) => {
      setHandData(data);
      checkButtonHover(data);
    });

    socketRef.current.on('game_state', (data) => {
      setGameState(data);
    });

    socketRef.current.on('game_started', (data) => {
      setGameState(data);
      setScreen('game');
    });

    socketRef.current.on('game_stopped', (data) => {
      setScreen('menu');
    });

    socketRef.current.on('debug_toggled', (data) => {
      setGameState(prev => ({ ...prev, debug_mode: data.debug_mode }));
    });

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);

  useEffect(() => {
    if (handData && handData.fist_closed && hoverIndex === 0) {
      setFistTimer(prev => {
        const newTimer = prev + 1;
        if (newTimer >= FIST_HOLD_FRAMES) {
          handleStartGame();
          return 0;
        }
        return newTimer;
      });
    } else {
      setFistTimer(0);
    }
  }, [handData, hoverIndex]);

  const checkButtonHover = (hand) => {
    if (!hand || hand.x === undefined) {
      setHoverIndex(null);
      return;
    }

    const buttonHeight = 80;
    const buttonWidth = 400;
    const buttonSpacing = 30;
    const startY = 200;
    const centerX = (gameState.frame_width / 2) - (buttonWidth / 2);

    for (let i = 0; i < buttons.length; i++) {
      const y = startY + i * (buttonHeight + buttonSpacing);
      if (hand.x >= centerX && hand.x <= centerX + buttonWidth &&
          hand.y >= y && hand.y <= y + buttonHeight) {
        setHoverIndex(i);
        return;
      }
    }
    setHoverIndex(null);
  };

  const handleStartGame = () => {
    if (socketRef.current) {
      socketRef.current.emit('start_game');
    }
  };

  const handleStopGame = () => {
    if (socketRef.current) {
      socketRef.current.emit('stop_game');
    }
  };

  const handleToggleDebug = () => {
    if (socketRef.current) {
      socketRef.current.emit('toggle_debug');
    }
  };

  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.key === 'd') {
        handleToggleDebug();
      } else if (e.key === 'Escape' && screen === 'game') {
        handleStopGame();
      } else if (e.key === 'q') {
        window.close();
      } else if (e.key === ' ' && screen === 'menu' && hoverIndex === 0) {
        e.preventDefault();
        handleStartGame();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [screen, hoverIndex]);

  if (!cameraReady) {
    return (
      <div style={{
        width: '100vw',
        height: '100vh',
        backgroundColor: '#000',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#FFF',
        fontSize: '24px'
      }}>
        Initializing camera...
      </div>
    );
  }

  return (
    <div style={{ 
      width: '100vw', 
      height: '100vh', 
      backgroundColor: '#000', 
      position: 'relative', 
      overflow: 'hidden',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      {/* Video Feed */}
      {videoFrame && (
        <img
          src={videoFrame}
          alt="Camera feed"
          style={{
            maxWidth: '100%',
            maxHeight: '100%',
            width: 'auto',
            height: 'auto',
            objectFit: 'contain'
          }}
        />
      )}

      {/* Menu Screen */}
      {screen === 'menu' && (
        <MenuScreen
          buttons={buttons}
          hoverIndex={hoverIndex}
          handData={handData}
          fistTimer={fistTimer}
          fistHoldFrames={FIST_HOLD_FRAMES}
          debugMode={gameState.debug_mode}
          onStartGame={handleStartGame}
          frameWidth={gameState.frame_width}
          frameHeight={gameState.frame_height}
        />
      )}

      {/* Game Screen */}
      {screen === 'game' && (
        <GameScreen
          gameState={gameState}
          onStopGame={handleStopGame}
        />
      )}
    </div>
  );
}

function MenuScreen({ buttons, hoverIndex, handData, fistTimer, fistHoldFrames, debugMode, onStartGame, frameWidth, frameHeight }) {
  const buttonHeight = 80;
  const buttonWidth = 400;
  const buttonSpacing = 30;

  return (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(20, 20, 20, 0.7)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      pointerEvents: 'none'
    }}>
      <h1 style={{
        color: '#00FFFF',
        fontSize: '48px',
        fontWeight: 'bold',
        marginBottom: '50px',
        textAlign: 'center',
        textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)'
      }}>
        BODY PART TOUCH GAME
      </h1>

      {debugMode && (
        <div style={{
          position: 'absolute',
          top: '20px',
          right: '20px',
          color: '#00FFFF',
          fontSize: '24px',
          fontWeight: 'bold',
          textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)'
        }}>
          DEBUG MODE: ON
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '30px', alignItems: 'center', pointerEvents: 'auto' }}>
        {buttons.map((button, index) => (
          <div key={index} style={{ position: 'relative' }}>
            <button
              onClick={() => index === 0 && onStartGame()}
              disabled={button.disabled}
              style={{
                width: `${buttonWidth}px`,
                height: `${buttonHeight}px`,
                backgroundColor: hoverIndex === index ? lighten(button.color) : button.color,
                border: `3px solid ${hoverIndex === index ? '#FFF' : '#C8C8C8'}`,
                color: '#FFF',
                fontSize: '24px',
                fontWeight: 'bold',
                cursor: button.disabled ? 'not-allowed' : 'pointer',
                opacity: button.disabled ? 0.6 : 1,
                borderRadius: '5px'
              }}
            >
              {button.text}
            </button>
            {button.disabled && (
              <span style={{
                position: 'absolute',
                left: '420px',
                top: '50%',
                transform: 'translateY(-50%)',
                color: '#969696',
                fontSize: '18px',
                whiteSpace: 'nowrap'
              }}>
                Coming Soon!
              </span>
            )}
            {index === 0 && hoverIndex === 0 && fistTimer > 0 && (
              <div style={{
                position: 'absolute',
                bottom: '-20px',
                left: 0,
                width: `${buttonWidth}px`,
                height: '10px',
                backgroundColor: 'rgba(255, 255, 255, 0.3)',
                border: '2px solid #FFF',
                borderRadius: '5px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${(fistTimer / fistHoldFrames) * 100}%`,
                  height: '100%',
                  backgroundColor: '#0F0',
                  transition: 'width 0.05s linear'
                }} />
              </div>
            )}
          </div>
        ))}
      </div>

      {handData && handData.x !== undefined && (
        <div style={{
          position: 'absolute',
          left: `${handData.x}px`,
          top: `${handData.y}px`,
          transform: 'translate(-50%, -50%)',
          pointerEvents: 'none',
          zIndex: 1000
        }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            backgroundColor: handData.fist_closed ? '#0F0' : '#FF00FF',
            border: '3px solid #FFF',
            boxShadow: '0 0 10px rgba(0, 0, 0, 0.5)'
          }} />
          <div style={{
            position: 'absolute',
            top: '0px',
            left: '50px',
            color: '#FFFF00',
            fontSize: '18px',
            fontWeight: 'bold',
            whiteSpace: 'nowrap',
            textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)'
          }}>
            {handData.fist_closed ? 'FIST!' : 'OPEN'}
          </div>
        </div>
      )}

      <div style={{
        position: 'absolute',
        bottom: '50px',
        left: '50px',
        color: '#C8C8C8',
        fontSize: '16px',
        textShadow: '1px 1px 2px rgba(0, 0, 0, 0.8)'
      }}>
        <div>Hover your hand over a button</div>
        <div>Close your fist, click mouse, or press SPACE to select</div>
        <div>Press 'd' to toggle debug mode</div>
      </div>
    </div>
  );
}

function GameScreen({ gameState, onStopGame }) {
  return (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      pointerEvents: 'none'
    }}>
      {gameState.target_pos && (
        <div style={{
          position: 'absolute',
          left: `${gameState.target_pos[0]}px`,
          top: `${gameState.target_pos[1]}px`,
          transform: 'translate(-50%, -50%)',
        }}>
          <div style={{
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            border: '3px solid #F00',
            boxShadow: '0 0 10px rgba(255, 0, 0, 0.5)'
          }} />
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            backgroundColor: '#F00'
          }} />
        </div>
      )}

      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        color: '#FFF',
        fontSize: '32px',
        fontWeight: 'bold',
        textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)'
      }}>
        Touch with: {gameState.target_body_part || 'Loading...'}
      </div>

      <div style={{
        position: 'absolute',
        top: '60px',
        left: '20px',
        color: '#0F0',
        fontSize: '32px',
        fontWeight: 'bold',
        textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)'
      }}>
        Score: {gameState.score}
      </div>

      {gameState.debug_mode && (
        <div style={{
          position: 'absolute',
          top: '100px',
          right: '20px',
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          padding: '20px',
          color: '#FFF',
          fontSize: '14px',
          fontFamily: 'monospace',
          borderRadius: '5px',
          border: '1px solid #00FFFF'
        }}>
          <div style={{ color: '#0FF', fontWeight: 'bold', marginBottom: '10px' }}>DEBUG MODE ON</div>
          <div>Target: {gameState.target_body_part || 'N/A'}</div>
          <div>Target Pos: ({gameState.target_pos?.[0] || 0}, {gameState.target_pos?.[1] || 0})</div>
          <div>Frame: {gameState.frame_width}x{gameState.frame_height}</div>
          <div>Active: {gameState.active ? 'Yes' : 'No'}</div>
        </div>
      )}

      <div style={{
        position: 'absolute',
        bottom: '20px',
        left: '20px',
        color: '#FFF',
        fontSize: '16px',
        textShadow: '2px 2px 4px rgba(0, 0, 0, 0.8)'
      }}>
        Press 'ESC' for menu, 'd' for debug, or 'q' to quit
      </div>
    </div>
  );
}

function lighten(color) {
  const hex = color.replace('#', '');
  const r = Math.min(255, parseInt(hex.substr(0, 2), 16) + 30);
  const g = Math.min(255, parseInt(hex.substr(2, 2), 16) + 30);
  const b = Math.min(255, parseInt(hex.substr(4, 2), 16) + 30);
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

export default App;