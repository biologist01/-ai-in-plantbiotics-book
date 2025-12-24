import React, { useState, useEffect, useRef, useCallback } from 'react';
import type {
  Room as LKRoom,
  LocalAudioTrack,
  RemoteTrack,
  RemoteTrackPublication,
  RemoteParticipant,
  Participant,
} from 'livekit-client';

// Use local backend in development, Railway in production
const BACKEND_URL = process.env.NODE_ENV === 'development'
  ? 'http://localhost:8000'
  : 'https://physical-ai-backend-production-1c69.up.railway.app';

interface VoiceChatProps {
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

export default function VoiceChat({ isOpen, onClose }: VoiceChatProps): React.ReactElement | null {
  const [voiceStatus, setVoiceStatus] = useState<'idle' | 'connecting' | 'connected' | 'speaking' | 'listening' | 'processing'>('idle');
  const [isMuted, setIsMuted] = useState(false);
  const [callDuration, setCallDuration] = useState(0);
  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
  const [currentTranscript, setCurrentTranscript] = useState<string>('');
  const [agentSpeaking, setAgentSpeaking] = useState(false);
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [agentAudioLevel, setAgentAudioLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [micWarning, setMicWarning] = useState<string | null>(null);

  const roomRef = useRef<LKRoom | null>(null);
  const localAudioTrackRef = useRef<LocalAudioTrack | null>(null);
  const remoteAudioContainerRef = useRef<HTMLDivElement>(null);
  const callStartTimeRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const remoteAnalyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const lowAudioCountRef = useRef<number>(0);

  // Scroll to bottom of transcripts
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcripts, currentTranscript]);

  // Call duration timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (voiceStatus === 'connected' || voiceStatus === 'speaking' || voiceStatus === 'listening' || voiceStatus === 'processing') {
      interval = setInterval(() => {
        if (callStartTimeRef.current) {
          setCallDuration(Math.floor((Date.now() - callStartTimeRef.current) / 1000));
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [voiceStatus]);

  // Audio level visualization with mic quality detection
  const startAudioVisualization = useCallback((stream: MediaStream, isLocal: boolean) => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }
      const audioContext = audioContextRef.current;
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.3; // Faster response to speech
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      
      if (isLocal) {
        analyserRef.current = analyser;
      } else {
        remoteAnalyserRef.current = analyser;
      }

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      
      const updateLevel = () => {
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        const normalizedLevel = Math.min(average / 128, 1);
        
        if (isLocal) {
          setAudioLevel(normalizedLevel);
          // Lower threshold for speech detection (0.05 instead of 0.1)
          const isSpeaking = normalizedLevel > 0.05;
          setUserSpeaking(isSpeaking);
          
          // Monitor for consistently low audio - potential mic issue
          if (normalizedLevel < 0.02) {
            lowAudioCountRef.current++;
            // If low audio for 5+ seconds (300 frames at 60fps), warn user
            if (lowAudioCountRef.current > 300) {
              setMicWarning('Mic level is very low. Try speaking louder or check your microphone settings.');
            }
          } else {
            lowAudioCountRef.current = 0;
            setMicWarning(null);
          }
        } else {
          setAgentAudioLevel(normalizedLevel);
          setAgentSpeaking(normalizedLevel > 0.1);
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
    lowAudioCountRef.current = 0;
    setMicWarning(null);
    setAudioLevel(0);
    setAgentAudioLevel(0);
  }, []);

  const clearRemoteAudio = useCallback(() => {
    if (remoteAudioContainerRef.current) {
      remoteAudioContainerRef.current.innerHTML = '';
    }
  }, []);

  const startVoiceCall = useCallback(async () => {
    if (voiceStatus !== 'idle') return;
    if (typeof window === 'undefined') return;

    setError(null);
    setVoiceStatus('connecting');
    setTranscripts([]);
    setCurrentTranscript('');

    try {
      const livekit = await import('livekit-client');
      const { Room, RoomEvent, Track, createLocalAudioTrack, ParticipantEvent } = livekit;

      // Request voice session from backend
      const sessionResponse = await fetch(`${BACKEND_URL}/voice/session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_name: 'web-user' }),
      });

      if (!sessionResponse.ok) {
        const text = await sessionResponse.text();
        throw new Error(`Failed to start voice session: ${text || sessionResponse.status}`);
      }

      const sessionData = (await sessionResponse.json()) as { room: string; token: string; url: string };

      const room = new Room({
        adaptiveStream: true,
        dynacast: true,
        // Optimized audio capture settings for better speech recognition
        audioCaptureDefaults: {
          echoCancellation: true,
          noiseSuppression: false, // Disable - can clip quiet speech and cause words to be missed
          autoGainControl: true,   // Keep - helps normalize volume levels
          channelCount: 1,         // Mono is better for speech
          sampleRate: 48000,       // Higher sample rate for better quality
          sampleSize: 16,          // 16-bit audio
        },
        audioOutput: {
          deviceId: 'default',
        },
        disconnectOnPageLeave: true,
      });

      // Handle remote audio tracks (agent's voice)
      room.on(
        RoomEvent.TrackSubscribed,
        (track: RemoteTrack, _pub: RemoteTrackPublication, participant: RemoteParticipant) => {
          if (track.kind !== Track.Kind.Audio) return;
          
          const el = track.attach() as HTMLAudioElement;
          el.autoplay = true;
          el.controls = false;
          el.setAttribute('data-lk-remote-audio', 'true');
          remoteAudioContainerRef.current?.appendChild(el);

          // Start visualization for remote audio
          const mediaStream = new MediaStream([track.mediaStreamTrack]);
          startAudioVisualization(mediaStream, false);
        }
      );

      room.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
        track.detach().forEach((el) => el.remove());
      });

      // Handle transcription events
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
        callStartTimeRef.current = null;
        setVoiceStatus('idle');
        setCallDuration(0);
      });

      room.on(RoomEvent.ParticipantConnected, (participant: RemoteParticipant) => {
        setTranscripts(prev => [
          ...prev,
          {
            id: `system-${Date.now()}`,
            role: 'assistant',
            text: '🎙️ AI Assistant connected. Start speaking!',
            timestamp: new Date(),
            isFinal: true,
          }
        ]);
      });

      await room.connect(sessionData.url, sessionData.token);
      roomRef.current = room;
      callStartTimeRef.current = Date.now();

      // Create and publish local audio track with optimized settings
      const localAudioTrack = await createLocalAudioTrack({
        echoCancellation: true,
        noiseSuppression: false,  // Disabled - browser noise suppression often clips speech
        autoGainControl: true,    // Helps with varying voice volumes
        channelCount: 1,          // Mono for speech
        sampleRate: 48000,        // High quality sample rate
      });

      // Set track to high priority for better transmission
      localAudioTrackRef.current = localAudioTrack;
      await room.localParticipant.publishTrack(localAudioTrack, {
        audioPreset: livekit.AudioPresets.speech,  // Optimized for speech
        dtx: true,  // Discontinuous transmission - saves bandwidth during silence
        red: true,  // Redundant encoding - helps recover lost packets
      });

      // Start visualization for local audio
      const localStream = new MediaStream([localAudioTrack.mediaStreamTrack]);
      startAudioVisualization(localStream, true);

      setVoiceStatus('connected');
    } catch (err) {
      console.error('Voice call error:', err);
      setError(err instanceof Error ? err.message : 'Failed to start voice call');
      setVoiceStatus('idle');
      await stopVoiceCall();
    }
  }, [voiceStatus, clearRemoteAudio, stopAudioVisualization, startAudioVisualization]);

  const stopVoiceCall = useCallback(async () => {
    try {
      const room = roomRef.current;
      const localTrack = localAudioTrackRef.current;

      if (room && localTrack) {
        try {
          room.localParticipant.unpublishTrack(localTrack);
        } catch { /* ignore */ }
        try {
          localTrack.stop();
        } catch { /* ignore */ }
      }

      if (room) {
        try {
          room.disconnect();
        } catch { /* ignore */ }
      }
    } finally {
      clearRemoteAudio();
      stopAudioVisualization();
      roomRef.current = null;
      localAudioTrackRef.current = null;
      callStartTimeRef.current = null;
      setVoiceStatus('idle');
      setCallDuration(0);
      setIsMuted(false);
    }
  }, [clearRemoteAudio, stopAudioVisualization]);

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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      void stopVoiceCall();
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, [stopVoiceCall]);

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        if (voiceStatus !== 'idle') {
          void stopVoiceCall();
        }
        onClose();
      }
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, voiceStatus, stopVoiceCall, onClose]);

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!isOpen) return null;

  const isConnected = voiceStatus === 'connected' || voiceStatus === 'speaking' || voiceStatus === 'listening' || voiceStatus === 'processing';

  return (
    <div className="voice-chat-overlay">
      <div className="voice-chat-container">
        {/* Animated Background */}
        <div className="voice-chat-bg">
          <div className="voice-orb voice-orb-1" style={{ transform: `scale(${1 + agentAudioLevel * 0.5})` }} />
          <div className="voice-orb voice-orb-2" style={{ transform: `scale(${1 + agentAudioLevel * 0.3})` }} />
          <div className="voice-orb voice-orb-3" style={{ transform: `scale(${1 + audioLevel * 0.4})` }} />
        </div>

        {/* Header */}
        <div className="voice-chat-header">
          <button className="voice-back-btn" onClick={() => { void stopVoiceCall(); onClose(); }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
          </button>
          <div className="voice-header-info">
            <h2>🌱 Plant AI Voice</h2>
            {isConnected && (
              <span className="voice-duration">{formatDuration(callDuration)}</span>
            )}
          </div>
          <div className="voice-status-indicator">
            <span className={`voice-status-dot ${isConnected ? 'active' : ''}`} />
            <span className="voice-status-text">
              {voiceStatus === 'idle' && 'Ready'}
              {voiceStatus === 'connecting' && 'Connecting...'}
              {isConnected && (agentSpeaking ? 'AI Speaking' : userSpeaking ? 'Listening...' : 'Connected')}
            </span>
          </div>
        </div>

        {/* Main Content */}
        <div className="voice-chat-main">
          {/* AI Avatar with Audio Visualization */}
          <div className="voice-avatar-container">
            <div className={`voice-avatar ${agentSpeaking ? 'speaking' : ''} ${isConnected ? 'connected' : ''}`}>
              <div className="voice-avatar-rings">
                <div className="voice-ring voice-ring-1" style={{ transform: `scale(${1 + agentAudioLevel * 0.8})`, opacity: 0.6 - agentAudioLevel * 0.3 }} />
                <div className="voice-ring voice-ring-2" style={{ transform: `scale(${1 + agentAudioLevel * 0.5})`, opacity: 0.4 - agentAudioLevel * 0.2 }} />
                <div className="voice-ring voice-ring-3" style={{ transform: `scale(${1 + agentAudioLevel * 0.3})`, opacity: 0.2 - agentAudioLevel * 0.1 }} />
              </div>
              <div className="voice-avatar-inner">
                <div className="voice-avatar-icon">
                  {voiceStatus === 'connecting' ? (
                    <div className="voice-loading-spinner" />
                  ) : (
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" fill="currentColor" opacity="0.2"/>
                      <path d="M8 14s1.5 2 4 2 4-2 4-2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                      <circle cx="9" cy="10" r="1.5" fill="currentColor"/>
                      <circle cx="15" cy="10" r="1.5" fill="currentColor"/>
                      <path d="M12 2v1M12 21v1M4.22 4.22l.7.7M18.36 18.36l.7.7M2 12h1M21 12h1M4.22 19.78l.7-.7M18.36 5.64l.7-.7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.3"/>
                    </svg>
                  )}
                </div>
                <div className="voice-avatar-label">
                  {voiceStatus === 'connecting' ? 'Connecting...' : 'Plant AI'}
                </div>
              </div>
            </div>
          </div>

          {/* Transcript Area */}
          <div className="voice-transcript-container">
            <div className="voice-transcript-header">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
              <span>Live Transcript</span>
            </div>
            <div className="voice-transcript-content">
              {error && (
                <div className="voice-transcript-error">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  {error}
                </div>
              )}
              {micWarning && (
                <div className="voice-transcript-warning">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                  </svg>
                  {micWarning}
                </div>
              )}
              {transcripts.length === 0 && !currentTranscript && !error && (
                <div className="voice-transcript-empty">
                  {isConnected 
                    ? "Start speaking — your conversation will appear here..."
                    : "Click the microphone to start a voice conversation with Plant AI"}
                </div>
              )}
              {transcripts.map((entry) => (
                <div key={entry.id} className={`voice-transcript-entry ${entry.role}`}>
                  <div className="voice-transcript-avatar">
                    {entry.role === 'assistant' ? '🌱' : '👤'}
                  </div>
                  <div className="voice-transcript-bubble">
                    <div className="voice-transcript-text">{entry.text}</div>
                    <div className="voice-transcript-time">
                      {entry.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                </div>
              ))}
              {currentTranscript && (
                <div className="voice-transcript-entry user current">
                  <div className="voice-transcript-avatar">👤</div>
                  <div className="voice-transcript-bubble">
                    <div className="voice-transcript-text">{currentTranscript}</div>
                    <div className="voice-transcript-typing">
                      <span></span><span></span><span></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={transcriptEndRef} />
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="voice-controls">
          {/* User Audio Level */}
          <div className="voice-user-level">
            <div className="voice-level-bars">
              {[...Array(5)].map((_, i) => (
                <div 
                  key={i} 
                  className={`voice-level-bar ${audioLevel > (i + 1) * 0.2 ? 'active' : ''}`}
                  style={{ height: `${20 + i * 8}px` }}
                />
              ))}
            </div>
            <span className="voice-level-label">{isMuted ? 'Muted' : 'Your Mic'}</span>
          </div>

          {/* Main Control Buttons */}
          <div className="voice-control-buttons">
            {/* Mute Button */}
            <button 
              className={`voice-control-btn voice-mute-btn ${isMuted ? 'muted' : ''}`}
              onClick={toggleMute}
              disabled={!isConnected}
              title={isMuted ? 'Unmute' : 'Mute'}
            >
              {isMuted ? (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="1" y1="1" x2="23" y2="23"/>
                  <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"/>
                  <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
              ) : (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
              )}
            </button>

            {/* Main Call Button */}
            <button 
              className={`voice-control-btn voice-call-btn ${isConnected ? 'active' : ''} ${voiceStatus === 'connecting' ? 'connecting' : ''}`}
              onClick={() => isConnected ? void stopVoiceCall() : void startVoiceCall()}
              disabled={voiceStatus === 'connecting'}
            >
              {voiceStatus === 'connecting' ? (
                <div className="voice-btn-spinner" />
              ) : isConnected ? (
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>
                  <line x1="1" y1="1" x2="23" y2="23" strokeWidth="3"/>
                </svg>
              ) : (
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
              )}
            </button>

            {/* Speaker/Volume Button */}
            <button 
              className="voice-control-btn voice-speaker-btn"
              disabled={!isConnected}
              title="Speaker"
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
              </svg>
            </button>
          </div>

          {/* Tips */}
          <div className="voice-tips">
            <span>💡 Tip: Ask about plant genetics, ML in agriculture, or any textbook topic!</span>
          </div>
        </div>

        {/* Hidden audio container */}
        <div ref={remoteAudioContainerRef} style={{ display: 'none' }} />
      </div>
    </div>
  );
}
