import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { api } from '@site/src/lib/auth';
import '@site/src/css/custom.css';

interface AuthModalProps {
  onClose: () => void;
  onSuccess: (user: any, token: string) => void;
}

export default function AuthModal({ onClose, onSuccess }: AuthModalProps): React.ReactElement | null {
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [mounted, setMounted] = useState(false);
  
  // Form state
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [backgroundType, setBackgroundType] = useState('software');
  const [learningGoals, setLearningGoals] = useState('');

  // Ensure we're on the client side for portal
  useEffect(() => {
    setMounted(true);
    // Prevent body scroll when modal is open
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (isLogin) {
        const response = await api.login({ email, password });
        localStorage.setItem('auth_token', response.access_token);
        localStorage.setItem('user', JSON.stringify(response.user));
        onSuccess(response.user, response.access_token);
      } else {
        const response = await api.signup({
          email,
          name,
          password,
          background_type: backgroundType,
          learning_goals: learningGoals || undefined,
        });
        localStorage.setItem('auth_token', response.access_token);
        localStorage.setItem('user', JSON.stringify(response.user));
        onSuccess(response.user, response.access_token);
      }
    } catch (err: any) {
      setError(err.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  const modalContent = (
    <div className="auth-modal-overlay" onClick={onClose}>
      <div className="auth-modal" onClick={(e) => e.stopPropagation()}>
        <button className="auth-modal-close" onClick={onClose}>✕</button>
        
        <h2>{isLogin ? 'Welcome Back!' : 'Create Account'}</h2>
        <p className="auth-modal-subtitle">
          {isLogin 
            ? 'Sign in to access personalized content' 
            : 'Join to get personalized learning experience'}
        </p>

        <form onSubmit={handleSubmit}>
          {!isLogin && (
            <div className="form-group">
              <label>Full Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                placeholder="John Doe"
              />
            </div>
          )}

          <div className="form-group">
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="you@example.com"
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="••••••••"
              minLength={6}
            />
          </div>

          {!isLogin && (
            <>
              <div className="form-group">
                <label>Your Background</label>
                <div className="background-radio-group">
                  <label className="radio-option">
                    <input
                      type="radio"
                      name="background"
                      value="software"
                      checked={backgroundType === 'software'}
                      onChange={(e) => setBackgroundType(e.target.value)}
                    />
                    <div className="radio-content">
                      <strong>💻 Software Developer</strong>
                      <span>I know programming but need help with hardware/robotics</span>
                    </div>
                  </label>
                  <label className="radio-option">
                    <input
                      type="radio"
                      name="background"
                      value="hardware"
                      checked={backgroundType === 'hardware'}
                      onChange={(e) => setBackgroundType(e.target.value)}
                    />
                    <div className="radio-content">
                      <strong>🔧 Hardware/Electronics Engineer</strong>
                      <span>I know electronics but need help with programming</span>
                    </div>
                  </label>
                </div>
              </div>

              <div className="form-group">
                <label>Learning Goals (Optional)</label>
                <textarea
                  value={learningGoals}
                  onChange={(e) => setLearningGoals(e.target.value)}
                  placeholder="What do you want to learn? e.g., Build autonomous robots, Learn ROS 2, etc."
                  rows={3}
                />
              </div>
            </>
          )}

          {error && <div className="error-message">{error}</div>}

          <button type="submit" className="auth-submit-btn" disabled={loading}>
            {loading ? 'Please wait...' : (isLogin ? 'Sign In' : 'Create Account')}
          </button>
        </form>

        <div className="auth-toggle">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button 
            type="button"
            onClick={() => {
              setIsLogin(!isLogin);
              setError('');
            }}
            className="auth-toggle-btn"
          >
            {isLogin ? 'Sign up' : 'Sign in'}
          </button>
        </div>
      </div>
    </div>
  );

  // Use portal to render modal at document body level
  if (!mounted) return null;
  
  return createPortal(modalContent, document.body);
}
