import React from 'react';
import { Bell, Lock, User, Shield } from 'lucide-react';

const Settings = () => {
    return (
        <div className="max-w-3xl mx-auto space-y-8">
            <div>
                <h1 className="text-2xl font-bold text-slate-800">Settings</h1>
                <p className="text-slate-500">Manage system preferences</p>
            </div>

            <div className="space-y-6">
                <Section title="Profile Settings" icon={<User size={20} />}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <Input label="Full Name" defaultValue="Admin User" />
                        <Input label="Email" defaultValue="admin@college.edu" />
                    </div>
                </Section>

                <Section title="Notifications" icon={<Bell size={20} />}>
                    <Toggle label="Email Alerts" description="Receive daily summary emails" checked />
                    <Toggle label="Real-time Notifications" description="Browser alerts for new detections" />
                </Section>

                <Section title="Security" icon={<Lock size={20} />}>
                    <button className="text-indigo-600 font-medium hover:underline">Change Password</button>
                </Section>

                <Section title="System" icon={<Shield size={20} />}>
                    <div className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                        <span className="text-sm font-medium text-slate-700">Camera Source</span>
                        <span className="text-sm text-slate-500">Default (0)</span>
                    </div>
                </Section>
            </div>
        </div>
    );
};

const Section = ({ title, icon, children }) => (
    <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
        <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-50">
            <div className="text-indigo-600">{icon}</div>
            <h2 className="text-lg font-bold text-slate-800">{title}</h2>
        </div>
        <div className="space-y-4">{children}</div>
    </div>
);

const Input = ({ label, defaultValue }) => (
    <div className="space-y-1">
        <label className="text-xs font-bold text-slate-500 uppercase">{label}</label>
        <input
            type="text"
            defaultValue={defaultValue}
            className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
    </div>
);

const Toggle = ({ label, description, checked }) => (
    <div className="flex items-center justify-between">
        <div>
            <p className="font-medium text-slate-800">{label}</p>
            <p className="text-xs text-slate-500">{description}</p>
        </div>
        <div className={`w-12 h-6 rounded-full p-1 cursor-pointer transition-colors ${checked ? 'bg-indigo-600' : 'bg-slate-200'}`}>
            <div className={`w-4 h-4 bg-white rounded-full shadow-sm transform transition-transform ${checked ? 'translate-x-6' : ''}`}></div>
        </div>
    </div>
);

export default Settings;
