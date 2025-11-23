import React from 'react';
import { LayoutDashboard, Users, FileText, Settings, LogOut } from 'lucide-react';

const Sidebar = () => {
    const menuItems = [
        { icon: <LayoutDashboard size={20} />, label: 'Dashboard', active: true },
        { icon: <Users size={20} />, label: 'Students', active: false },
        { icon: <FileText size={20} />, label: 'Reports', active: false },
        { icon: <Settings size={20} />, label: 'Settings', active: false },
    ];

    return (
        <div className="w-64 h-screen bg-white border-r border-slate-100 flex flex-col fixed left-0 top-0 z-50">
            {/* Brand Logo */}
            <div className="p-8 flex items-center gap-3">
                <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-xl">A</span>
                </div>
                <span className="text-2xl font-bold text-slate-800 tracking-tight">Attendify</span>
            </div>

            {/* Navigation */}
            <nav className="flex-1 px-4 space-y-2">
                <p className="px-4 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">Menu</p>
                {menuItems.map((item, index) => (
                    <button
                        key={index}
                        className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group ${item.active
                                ? 'bg-indigo-50 text-indigo-600 font-semibold shadow-sm'
                                : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900'
                            }`}
                    >
                        <span className={`${item.active ? 'text-indigo-600' : 'text-slate-400 group-hover:text-slate-600'}`}>
                            {item.icon}
                        </span>
                        {item.label}
                    </button>
                ))}
            </nav>

            {/* User Profile / Logout */}
            <div className="p-4 border-t border-slate-100">
                <button className="w-full flex items-center gap-3 px-4 py-3 text-slate-500 hover:text-red-500 hover:bg-red-50 rounded-xl transition-colors">
                    <LogOut size={20} />
                    <span className="font-medium">Logout</span>
                </button>
            </div>
        </div>
    );
};

export default Sidebar;
