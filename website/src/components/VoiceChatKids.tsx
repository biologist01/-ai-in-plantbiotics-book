/**
 * ═══════════════════════════════════════════════════════════════════════════
 * 🌱 PLANT BUDDY - Voice Chat for Kids (Ages 9-11)
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * DESIGN PHILOSOPHY:
 * - Zero cognitive load: Kids should understand instantly without instructions
 * - Gamified: Feels like playing a game, not using an app
 * - Character-driven: Friendly plant mascot creates emotional connection
 * - Reward-based: Every interaction feels like an achievement
 * - Visual-first: Animations speak louder than text
 * 
 * TARGET USERS:
 * - Age: 9-11 years (4th-5th grade)
 * - Short attention spans, attracted to colors and animations
 * - Love cartoons, games, and talking characters
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import '@site/src/css/voice-chat-kids.css';
import type {
  Room as LKRoom,
  LocalAudioTrack,
  RemoteTrack,
  RemoteTrackPublication,
  RemoteParticipant,
  Participant,
} from 'livekit-client';

// Backend URL configuration
const BACKEND_URL = process.env.NODE_ENV === 'development'
  ? 'http://localhost:8000'
  : 'https://physical-ai-backend-production-1c69.up.railway.app';

interface VoiceChatKidsProps {
  isOpen: boolean;
  onClose: () => void;
}

interface TranscriptEntry {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  timestamp: Date;
  isFinal: boolean;
}

// 🎉 Fun encouragement messages that make kids feel special
const ENCOURAGEMENTS = [
  "Wow! Great question! 🎉",
  "You're so smart! 🌟",
  "Super curious explorer! 🔭",
  "Amazing thinking! 💡",
  "You're a plant genius! 🌱",
  "Fantastic question! ✨",
  "Way to go! 🚀",
  "Brilliant! Keep asking! 🎯",
];

// 🎨 Fun "thinking" messages to keep kids engaged during processing
const THINKING_MESSAGES = [
  "Plant Buddy is thinking... 🌱",
  "Hmm, let me check my plant brain... 🧠",
  "Searching my garden of knowledge... 🌻",
  "Growing an answer for you... 🌿",
  "Digging up cool facts... 🔍",
];

export default function VoiceChatKids({ isOpen, onClose }: VoiceChatKidsProps): React.ReactElement | null {
  // Voice state management
  const [voiceStatus, setVoiceStatus] = useState<'idle' | 'connecting' | 'connected' | 'speaking' | 'listening' | 'processing'>('idle');
  const [isMuted, setIsMuted] = useState(false);
  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
  const [currentTranscript, setCurrentTranscript] = useState<string>('');
  const [agentSpeaking, setAgentSpeaking] = useState(false);
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [agentAudioLevel, setAgentAudioLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  
  // 🎮 Gamification states
  const [stars, setStars] = useState(0);
  const [currentEncouragement, setCurrentEncouragement] = useState<string | null>(null);
  const [showStarBurst, setShowStarBurst] = useState(false);
  const [thinkingMessage, setThinkingMessage] = useState(THINKING_MESSAGES[0]);
  const [mascotMood, setMascotMood] = useState<'happy' | 'excited' | 'thinking' | 'listening' | 'sleeping'>('sleeping');

  // Stable display state (debounced) - prevents flickering
  const [stableStatus, setStableStatus] = useState<'idle' | 'listening' | 'speaking' | 'thinking'>('idle');
  
  // Refs for audio handling
  const roomRef = useRef<LKRoom | null>(null);
  const localAudioTrackRef = useRef<LocalAudioTrack | null>(null);
  const remoteAudioContainerRef = useRef<HTMLDivElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const remoteAnalyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  // Debouncing refs to prevent flickering status changes
  const userSpeakingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const agentSpeakingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastUserSpeakingRef = useRef(false);
  const lastAgentSpeakingRef = useRef(false);

  // 🌟 Award a star when kid asks a question (gamification!)
  const awardStar = useCallback(() => {
    setStars(prev => prev + 1);
    setShowStarBurst(true);
    setCurrentEncouragement(ENCOURAGEMENTS[Math.floor(Math.random() * ENCOURAGEMENTS.length)]);
    
    // Clear effects after animation
    setTimeout(() => {
      setShowStarBurst(false);
      setCurrentEncouragement(null);
    }, 2500);
  }, []);

  // Update thinking message periodically to keep it fun
  useEffect(() => {
    if (voiceStatus === 'processing' || (agentSpeaking === false && voiceStatus === 'connected')) {
      const interval = setInterval(() => {
        setThinkingMessage(THINKING_MESSAGES[Math.floor(Math.random() * THINKING_MESSAGES.length)]);
      }, 2000);
      return () => clearInterval(interval);
    }
  }, [voiceStatus, agentSpeaking]);

  // Debounced stable status update - prevents rapid flickering
  useEffect(() => {
    const isConnected = voiceStatus === 'connected' || voiceStatus === 'speaking' || voiceStatus === 'listening' || voiceStatus === 'processing';
    
    if (voiceStatus === 'idle') {
      setStableStatus('idle');
    } else if (voiceStatus === 'connecting') {
      setStableStatus('thinking');
    } else if (isConnected) {
      // Use a simple priority system with minimum hold time
      if (agentSpeaking) {
        setStableStatus('speaking');
      } else if (userSpeaking) {
        setStableStatus('listening');
      } else {
        // Don't change immediately when both stop - wait a moment
        const timeout = setTimeout(() => {
          if (!lastAgentSpeakingRef.current && !lastUserSpeakingRef.current) {
            setStableStatus('thinking');
          }
        }, 800); // 800ms delay before showing "thinking"
        return () => clearTimeout(timeout);
      }
    }
  }, [voiceStatus, agentSpeaking, userSpeaking]);

  // Update mascot mood based on STABLE status (not raw audio)
  useEffect(() => {
    if (voiceStatus === 'idle') {
      setMascotMood('sleeping');
    } else if (voiceStatus === 'connecting') {
      setMascotMood('excited');
    } else if (stableStatus === 'speaking') {
      setMascotMood('happy');
    } else if (stableStatus === 'listening') {
      setMascotMood('listening');
    } else {
      setMascotMood('thinking');
    }
  }, [voiceStatus, stableStatus]);

  // Audio visualization
  const startAudioVisualization = useCallback((stream: MediaStream, isLocal: boolean) => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }
      const audioContext = audioContextRef.current;
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.3;
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      if (isLocal) {
        analyserRef.current = analyser;
      } else {
        remoteAnalyserRef.current = analyser;
      }

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      
      // Debounced speaking detection to prevent flickering
      let speakingFrameCount = 0;
      let silentFrameCount = 0;
      const SPEAKING_THRESHOLD = isLocal ? 0.05 : 0.1;
      const FRAMES_TO_START_SPEAKING = 3;  // Need 3 frames of sound to trigger "speaking"
      const FRAMES_TO_STOP_SPEAKING = 15;  // Need 15 frames of silence to stop "speaking"
      
      const updateLevel = () => {
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        const normalizedLevel = Math.min(average / 128, 1);
        
        const isSoundDetected = normalizedLevel > SPEAKING_THRESHOLD;
        
        if (isLocal) {
          setAudioLevel(normalizedLevel);
          
          // Debounced user speaking detection
          if (isSoundDetected) {
            speakingFrameCount++;
            silentFrameCount = 0;
            if (speakingFrameCount >= FRAMES_TO_START_SPEAKING && !lastUserSpeakingRef.current) {
              lastUserSpeakingRef.current = true;
              setUserSpeaking(true);
            }
          } else {
            silentFrameCount++;
            speakingFrameCount = 0;
            if (silentFrameCount >= FRAMES_TO_STOP_SPEAKING && lastUserSpeakingRef.current) {
              lastUserSpeakingRef.current = false;
              setUserSpeaking(false);
            }
          }
        } else {
          setAgentAudioLevel(normalizedLevel);
          
          // Debounced agent speaking detection
          if (isSoundDetected) {
            speakingFrameCount++;
            silentFrameCount = 0;
            if (speakingFrameCount >= FRAMES_TO_START_SPEAKING && !lastAgentSpeakingRef.current) {
              lastAgentSpeakingRef.current = true;
              setAgentSpeaking(true);
            }
          } else {
            silentFrameCount++;
            speakingFrameCount = 0;
            if (silentFrameCount >= FRAMES_TO_STOP_SPEAKING && lastAgentSpeakingRef.current) {
              lastAgentSpeakingRef.current = false;
              setAgentSpeaking(false);
            }
          }
        }
        
        animationFrameRef.current = requestAnimationFrame(updateLevel);
      };
      updateLevel();
    } catch (err) {
      console.error('Audio visualization error:', err);
    }
  }, []);

  const stopAudioVisualization = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    setAudioLevel(0);
    setAgentAudioLevel(0);
  }, []);

  const clearRemoteAudio = useCallback(() => {
    if (remoteAudioContainerRef.current) {
      remoteAudioContainerRef.current.innerHTML = '';
    }
  }, []);

  // Start voice call
  const startVoiceCall = useCallback(async () => {
    if (voiceStatus !== 'idle') return;
    if (typeof window === 'undefined') return;

    setError(null);
    setVoiceStatus('connecting');
    setTranscripts([]);
    setCurrentTranscript('');

    try {
      const livekit = await import('livekit-client');
      const { Room, RoomEvent, Track, createLocalAudioTrack } = livekit;

      const sessionResponse = await fetch(`${BACKEND_URL}/voice/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_name: 'young-explorer' }),
      });

      if (!sessionResponse.ok) {
        throw new Error('Oops! Could not connect. Try again! 🔄');
      }

      const sessionData = (await sessionResponse.json()) as { room: string; token: string; url: string };

      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
        audioCaptureDefaults: {
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: true,
          channelCount: 1,
          sampleRate: 48000,
          sampleSize: 16,
        },
        disconnectOnPageLeave: true,
      });

      room.on(
        RoomEvent.TrackSubscribed,
        (track: RemoteTrack, _pub: RemoteTrackPublication, participant: RemoteParticipant) => {
          if (track.kind !== Track.Kind.Audio) return;
          
          const el = track.attach() as HTMLAudioElement;
          el.autoplay = true;
          el.controls = false;
          remoteAudioContainerRef.current?.appendChild(el);

          const mediaStream = new MediaStream([track.mediaStreamTrack]);
          startAudioVisualization(mediaStream, false);
        }
      );

      room.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
        track.detach().forEach((el) => el.remove());
      });

      room.on(RoomEvent.TranscriptionReceived, (segments: any, participant?: Participant) => {
        for (const segment of segments) {
          const isAgent = participant?.identity !== room.localParticipant.identity;
          const role = isAgent ? 'assistant' : 'user';
          
          if (segment.final) {
            setTranscripts(prev => [
              ...prev,
              {
                id: segment.id || `${Date.now()}-${Math.random()}`,
                role,
                text: segment.text,
                timestamp: new Date(),
                isFinal: true,
              }
            ]);
            setCurrentTranscript('');
            
            // Award star when user finishes a question!
            if (role === 'user') {
              awardStar();
            }
          } else {
            setCurrentTranscript(segment.text);
          }
        }
      });

      room.on(RoomEvent.Disconnected, () => {
        clearRemoteAudio();
        stopAudioVisualization();
        roomRef.current = null;
        localAudioTrackRef.current = null;
        setVoiceStatus('idle');
      });

      room.on(RoomEvent.ParticipantConnected, () => {
        setTranscripts(prev => [
          ...prev,
          {
            id: `welcome-${Date.now()}`,
            role: 'assistant',
            text: 'Hi there, explorer! 🌱 Ask me anything about plants!',
            timestamp: new Date(),
            isFinal: true,
          }
        ]);
      });

      await room.connect(sessionData.url, sessionData.token);
      roomRef.current = room;

      const localAudioTrack = await createLocalAudioTrack({
        echoCancellation: true,
        noiseSuppression: false,
        autoGainControl: true,
        channelCount: 1,
        sampleRate: 48000,
      });

      localAudioTrackRef.current = localAudioTrack;
      await room.localParticipant.publishTrack(localAudioTrack, {
        audioPreset: livekit.AudioPresets.speech,
        dtx: true,
        red: true,
      });

      const localStream = new MediaStream([localAudioTrack.mediaStreamTrack]);
      startAudioVisualization(localStream, true);

      setVoiceStatus('connected');
    } catch (err) {
      console.error('Voice call error:', err);
      setError('Oops! Something went wrong. Tap to try again! 🔄');
      setVoiceStatus('idle');
    }
  }, [voiceStatus, clearRemoteAudio, stopAudioVisualization, startAudioVisualization, awardStar]);

  // Stop voice call
  const stopVoiceCall = useCallback(async () => {
    try {
      const room = roomRef.current;
      const localTrack = localAudioTrackRef.current;

      if (room && localTrack) {
        try { room.localParticipant.unpublishTrack(localTrack); } catch { /* ignore */ }
        try { localTrack.stop(); } catch { /* ignore */ }
      }
      if (room) {
        try { room.disconnect(); } catch { /* ignore */ }
      }
    } finally {
      clearRemoteAudio();
      stopAudioVisualization();
      roomRef.current = null;
      localAudioTrackRef.current = null;
      setVoiceStatus('idle');
      setIsMuted(false);
    }
  }, [clearRemoteAudio, stopAudioVisualization]);

  // Toggle mute
  const toggleMute = useCallback(() => {
    const track = localAudioTrackRef.current;
    if (track) {
      if (isMuted) {
        track.unmute();
      } else {
        track.mute();
      }
      setIsMuted(!isMuted);
    }
  }, [isMuted]);

  // Cleanup
  useEffect(() => {
    return () => {
      void stopVoiceCall();
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [stopVoiceCall]);

  // Handle close
  const handleClose = useCallback(() => {
    void stopVoiceCall();
    onClose();
  }, [stopVoiceCall, onClose]);

  if (!isOpen) return null;

  const isConnected = voiceStatus === 'connected' || voiceStatus === 'speaking' || voiceStatus === 'listening' || voiceStatus === 'processing';

  return (
    <div className="kids-voice-overlay">
      {/* ═══════════════════════════════════════════════════════════════════
          ANIMATED BACKGROUND
          Why: Colorful, moving backgrounds capture attention and create 
          a magical, immersive world that kids want to explore
      ═══════════════════════════════════════════════════════════════════ */}
      <div className="kids-bg">
        <div className="kids-bg-gradient" />
        <div className="kids-floating-shapes">
          {/* Floating leaves, stars, and sparkles */}
          <div className="kids-float kids-float-1">🍃</div>
          <div className="kids-float kids-float-2">✨</div>
          <div className="kids-float kids-float-3">🌿</div>
          <div className="kids-float kids-float-4">⭐</div>
          <div className="kids-float kids-float-5">🌱</div>
          <div className="kids-float kids-float-6">💫</div>
        </div>
        {/* Bubble animations for magical effect */}
        <div className="kids-bubbles">
          {[...Array(8)].map((_, i) => (
            <div key={i} className={`kids-bubble kids-bubble-${i + 1}`} />
          ))}
        </div>
      </div>

      <div className="kids-voice-container">
        {/* ═══════════════════════════════════════════════════════════════════
            HEADER WITH STARS COUNTER (Gamification)
            Why: Shows progress, makes kids feel accomplished, and 
            motivates them to keep asking questions
        ═══════════════════════════════════════════════════════════════════ */}
        <div className="kids-header">
          <button className="kids-close-btn" onClick={handleClose} aria-label="Go back">
            <span className="kids-close-icon">✕</span>
          </button>
          
          <div className="kids-title">
            <span className="kids-title-icon">🌱</span>
            <span className="kids-title-text">Plant Buddy</span>
          </div>
          
          {/* Star counter - gamification element */}
          <div className="kids-stars-counter">
            <span className="kids-star-icon">⭐</span>
            <span className="kids-star-count">{stars}</span>
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════
            MAIN MASCOT AREA
            Why: Central animated character creates emotional connection
            Kids talk TO the character, not to an app
        ═══════════════════════════════════════════════════════════════════ */}
        <div className="kids-mascot-area">
          {/* Star burst animation when kid earns a star */}
          {showStarBurst && (
            <div className="kids-star-burst">
              {[...Array(12)].map((_, i) => (
                <div key={i} className="kids-star-particle" style={{ '--i': i } as React.CSSProperties}>⭐</div>
              ))}
            </div>
          )}

          {/* Encouragement popup */}
          {currentEncouragement && (
            <div className="kids-encouragement">
              {currentEncouragement}
            </div>
          )}

          {/* ═══════════════════════════════════════════════════════════════
              PLANT BUDDY MASCOT
              - Animated character with different moods
              - Eyes blink, body bounces, reacts to voice
              - Makes the AI feel alive and friendly
          ═══════════════════════════════════════════════════════════════ */}
          <div className={`kids-mascot kids-mascot-${mascotMood}`}>
            {/* Sound waves when speaking */}
            {agentSpeaking && (
              <div className="kids-sound-waves">
                {[...Array(3)].map((_, i) => (
                  <div 
                    key={i} 
                    className="kids-sound-wave"
                    style={{ 
                      animationDelay: `${i * 0.15}s`,
                      transform: `scale(${1 + agentAudioLevel * (i + 1) * 0.3})`
                    }}
                  />
                ))}
              </div>
            )}

            {/* Listening sparkles */}
            {userSpeaking && isConnected && (
              <div className="kids-listening-sparkles">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="kids-sparkle" style={{ '--delay': `${i * 0.2}s` } as React.CSSProperties}>✨</div>
                ))}
              </div>
            )}

            {/* The pot (base) */}
            <div className="kids-mascot-pot">
              <div className="kids-mascot-pot-rim" />
              <div className="kids-mascot-pot-body" />
              <div className="kids-mascot-pot-soil" />
            </div>

            {/* The plant body */}
            <div className="kids-mascot-body">
              {/* Stem */}
              <div className="kids-mascot-stem" />
              
              {/* Leaves */}
              <div className="kids-mascot-leaves">
                <div className="kids-mascot-leaf kids-mascot-leaf-left" />
                <div className="kids-mascot-leaf kids-mascot-leaf-right" />
              </div>

              {/* Face */}
              <div className="kids-mascot-face">
                {/* Eyes */}
                <div className="kids-mascot-eyes">
                  <div className={`kids-mascot-eye kids-mascot-eye-left ${mascotMood === 'sleeping' ? 'closed' : ''}`}>
                    <div className="kids-mascot-pupil" />
                    <div className="kids-mascot-eye-shine" />
                  </div>
                  <div className={`kids-mascot-eye kids-mascot-eye-right ${mascotMood === 'sleeping' ? 'closed' : ''}`}>
                    <div className="kids-mascot-pupil" />
                    <div className="kids-mascot-eye-shine" />
                  </div>
                </div>

                {/* Mouth - changes based on mood */}
                <div className={`kids-mascot-mouth kids-mascot-mouth-${mascotMood}`}>
                  {agentSpeaking && <div className="kids-mascot-mouth-speaking" />}
                </div>

                {/* Blush cheeks */}
                <div className="kids-mascot-cheeks">
                  <div className="kids-mascot-cheek kids-mascot-cheek-left" />
                  <div className="kids-mascot-cheek kids-mascot-cheek-right" />
                </div>
              </div>
            </div>

            {/* Flower on top */}
            <div className="kids-mascot-flower">
              <div className="kids-mascot-flower-center" />
              {[...Array(6)].map((_, i) => (
                <div 
                  key={i} 
                  className="kids-mascot-petal"
                  style={{ '--rotation': `${i * 60}deg` } as React.CSSProperties}
                />
              ))}
            </div>
          </div>

          {/* Status text under mascot - uses STABLE status to prevent flickering */}
          <div className="kids-status-text">
            {voiceStatus === 'idle' && (
              <span className="kids-status-idle">Tap the button to wake me up! 👇</span>
            )}
            {voiceStatus === 'connecting' && (
              <span className="kids-status-connecting">
                <span className="kids-bounce-dots">
                  <span>.</span><span>.</span><span>.</span>
                </span>
                Waking up...
              </span>
            )}
            {isConnected && stableStatus === 'thinking' && (
              <span className="kids-status-ready">I'm listening! Ask me anything! 🎤</span>
            )}
            {isConnected && stableStatus === 'listening' && (
              <span className="kids-status-listening">
                <span className="kids-listening-dots">
                  <span></span><span></span><span></span>
                </span>
                Hearing you...
              </span>
            )}
            {isConnected && stableStatus === 'speaking' && (
              <span className="kids-status-speaking">Plant Buddy is talking... 🗣️</span>
            )}
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════
            🎪 SUPER COOL LIVE TRANSCRIPT THEATER
            Comic-book style speech bubbles with animations, particles,
            and visual effects that make conversations feel MAGICAL!
        ═══════════════════════════════════════════════════════════════════ */}
        <div className="kids-transcript-theater">
          {/* Decorative corner elements */}
          <div className="kids-theater-corner kids-theater-corner-tl">✨</div>
          <div className="kids-theater-corner kids-theater-corner-tr">🌟</div>
          <div className="kids-theater-corner kids-theater-corner-bl">💫</div>
          <div className="kids-theater-corner kids-theater-corner-br">⭐</div>
          
          {/* Theater header */}
          <div className="kids-theater-header">
            <div className="kids-theater-title">
              <span className="kids-theater-icon">📜</span>
              <span>Our Chat Story</span>
            </div>
            <div className="kids-theater-live-badge">
              <span className="kids-live-dot"></span>
              <span>LIVE</span>
            </div>
          </div>

          {/* Scrollable message area */}
          <div className="kids-theater-scroll">
            {error && (
              <div className="kids-comic-error">
                <div className="kids-error-character">😅</div>
                <div className="kids-error-bubble">
                  <span>{error}</span>
                  <div className="kids-bubble-tail kids-tail-left"></div>
                </div>
              </div>
            )}

            {transcripts.length === 0 && !currentTranscript && !error && (
              <div className="kids-empty-theater">
                <div className="kids-empty-icon">💬</div>
                <div className="kids-empty-text">
                  {isConnected 
                    ? "Start talking! Your words will appear here like magic! ✨"
                    : "Tap the green button to start our adventure! 🚀"
                  }
                </div>
              </div>
            )}
            
            {transcripts.slice(-4).map((entry, index) => (
              <div 
                key={entry.id} 
                className={`kids-comic-panel kids-comic-${entry.role}`}
                style={{ '--delay': `${index * 0.1}s` } as React.CSSProperties}
              >
                {/* Character avatar with expression */}
                <div className="kids-comic-character">
                  <div className={`kids-character-avatar kids-avatar-${entry.role}`}>
                    {entry.role === 'assistant' ? (
                      <div className="kids-plant-mini">
                        <div className="kids-plant-mini-pot">🪴</div>
                      </div>
                    ) : (
                      <div className="kids-kid-avatar">👧</div>
                    )}
                  </div>
                  <div className="kids-character-name">
                    {entry.role === 'assistant' ? 'Plant Buddy' : 'You'}
                  </div>
                </div>

                {/* Comic speech bubble */}
                <div className="kids-comic-bubble-container">
                  <div className={`kids-comic-bubble kids-bubble-${entry.role}`}>
                    <div className="kids-bubble-content">
                      {entry.text}
                    </div>
                    <div className="kids-bubble-sparkles">
                      {entry.role === 'assistant' && (
                        <>
                          <span className="kids-mini-sparkle">✨</span>
                          <span className="kids-mini-sparkle">🌿</span>
                        </>
                      )}
                    </div>
                    {/* Comic bubble tail */}
                    <div className={`kids-bubble-tail kids-tail-${entry.role}`}></div>
                  </div>
                  {/* Timestamp */}
                  <div className="kids-bubble-time">
                    {entry.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </div>
            ))}

            {/* Live typing indicator - SUPER animated */}
            {currentTranscript && (
              <div className="kids-comic-panel kids-comic-user kids-comic-live">
                <div className="kids-comic-character">
                  <div className="kids-character-avatar kids-avatar-user kids-avatar-talking">
                    <div className="kids-kid-avatar">👧</div>
                    <div className="kids-talking-waves">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                  <div className="kids-character-name">You</div>
                </div>
                <div className="kids-comic-bubble-container">
                  <div className="kids-comic-bubble kids-bubble-user kids-bubble-live">
                    <div className="kids-bubble-content">
                      {currentTranscript}
                      <span className="kids-live-cursor">|</span>
                    </div>
                    <div className="kids-live-indicator">
                      <span className="kids-wave-bar"></span>
                      <span className="kids-wave-bar"></span>
                      <span className="kids-wave-bar"></span>
                      <span className="kids-wave-bar"></span>
                      <span className="kids-wave-bar"></span>
                    </div>
                    <div className="kids-bubble-tail kids-tail-user"></div>
                  </div>
                </div>
              </div>
            )}

            {/* AI thinking indicator */}
            {isConnected && !agentSpeaking && !userSpeaking && transcripts.length > 0 && (
              <div className="kids-thinking-panel">
                <div className="kids-thinking-avatar">🌱</div>
                <div className="kids-thinking-bubble">
                  <div className="kids-thinking-dots">
                    <span></span><span></span><span></span>
                  </div>
                  <span className="kids-thinking-text">{thinkingMessage}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ═══════════════════════════════════════════════════════════════════
            BIG MICROPHONE BUTTON
            Why: ONE big button = zero confusion
            Pulsing animation invites interaction
            Kids know exactly what to tap
        ═══════════════════════════════════════════════════════════════════ */}
        <div className="kids-controls">
          {/* Audio level indicator - fun visual feedback */}
          {isConnected && (
            <div className="kids-audio-feedback">
              {[...Array(5)].map((_, i) => (
                <div 
                  key={i}
                  className={`kids-audio-bar ${audioLevel > (i + 1) * 0.18 ? 'active' : ''}`}
                  style={{ height: `${16 + i * 6}px` }}
                />
              ))}
            </div>
          )}

          {/* Main microphone button */}
          <button
            className={`kids-mic-button ${isConnected ? 'active' : ''} ${voiceStatus === 'connecting' ? 'connecting' : ''} ${userSpeaking ? 'speaking' : ''}`}
            onClick={() => isConnected ? void stopVoiceCall() : void startVoiceCall()}
            disabled={voiceStatus === 'connecting'}
            aria-label={isConnected ? 'Stop talking' : 'Start talking'}
          >
            {/* Pulsing rings when listening */}
            {isConnected && !isMuted && (
              <div className="kids-mic-rings">
                <div className="kids-mic-ring" style={{ transform: `scale(${1 + audioLevel * 0.5})` }} />
                <div className="kids-mic-ring" style={{ transform: `scale(${1.2 + audioLevel * 0.3})`, animationDelay: '0.2s' }} />
                <div className="kids-mic-ring" style={{ transform: `scale(${1.4 + audioLevel * 0.2})`, animationDelay: '0.4s' }} />
              </div>
            )}

            <div className="kids-mic-inner">
              {voiceStatus === 'connecting' ? (
                <div className="kids-mic-loading">
                  <div className="kids-spinner">🌱</div>
                </div>
              ) : isConnected ? (
                <div className="kids-mic-icon kids-mic-active">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor"/>
                    <path d="M5 10V12C5 15.866 8.13401 19 12 19C15.866 19 19 15.866 19 12V10" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                    <path d="M12 19V22" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                    <path d="M8 22H16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                  </svg>
                </div>
              ) : (
                <div className="kids-mic-icon">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor"/>
                    <path d="M5 10V12C5 15.866 8.13401 19 12 19C15.866 19 19 15.866 19 12V10" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                    <path d="M12 19V22" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                    <path d="M8 22H16" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
                  </svg>
                </div>
              )}
            </div>

            <span className="kids-mic-label">
              {isConnected ? 'Tap to Stop' : 'Tap to Talk!'}
            </span>
          </button>

          {/* Mute button - smaller, secondary */}
          {isConnected && (
            <button 
              className={`kids-mute-button ${isMuted ? 'muted' : ''}`}
              onClick={toggleMute}
              aria-label={isMuted ? 'Unmute' : 'Mute'}
            >
              {isMuted ? '🔇' : '🔊'}
            </button>
          )}
        </div>

        {/* Fun tip at bottom */}
        <div className="kids-fun-tip">
          <span className="kids-tip-icon">💡</span>
          <span className="kids-tip-text">
            {isConnected 
              ? "Ask about plants, flowers, or trees!"
              : "Ready for a plant adventure?"
            }
          </span>
        </div>
      </div>

      {/* Hidden audio container */}
      <div ref={remoteAudioContainerRef} style={{ display: 'none' }} />
    </div>
  );
}
