-- Appointment Scheduling Database Schema
-- This script creates tables for intelligent appointment scheduling functionality

-- Create the associates table for storing associate information
CREATE TABLE IF NOT EXISTS associates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    specialization TEXT NOT NULL DEFAULT 'Real Estate',
    availability_hours TEXT NOT NULL DEFAULT '9:00 AM - 6:00 PM',
    timezone TEXT NOT NULL DEFAULT 'America/New_York',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for associates table
CREATE INDEX IF NOT EXISTS idx_associates_email ON associates(email);
CREATE INDEX IF NOT EXISTS idx_associates_active ON associates(is_active);
CREATE INDEX IF NOT EXISTS idx_associates_specialization ON associates(specialization);

-- Create the appointments table for storing appointment details
CREATE TABLE IF NOT EXISTS appointments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    associate_id UUID NOT NULL REFERENCES associates(id) ON DELETE CASCADE,
    user_name TEXT NOT NULL,
    user_email TEXT NOT NULL,
    user_phone TEXT,
    scheduled_time TIMESTAMP WITH TIME ZONE NOT NULL,
    appointment_type TEXT NOT NULL DEFAULT 'consultation',
    status TEXT NOT NULL DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'confirmed', 'cancelled', 'completed', 'no_show')),
    notes TEXT,
    calendar_event_id TEXT,
    reminder_sent BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for appointments table
CREATE INDEX IF NOT EXISTS idx_appointments_session_id ON appointments(session_id);
CREATE INDEX IF NOT EXISTS idx_appointments_associate_id ON appointments(associate_id);
CREATE INDEX IF NOT EXISTS idx_appointments_user_email ON appointments(user_email);
CREATE INDEX IF NOT EXISTS idx_appointments_scheduled_time ON appointments(scheduled_time);
CREATE INDEX IF NOT EXISTS idx_appointments_status ON appointments(status);
CREATE INDEX IF NOT EXISTS idx_appointments_created_at ON appointments(created_at);

-- Create the availability_slots table for managing associate availability
CREATE TABLE IF NOT EXISTS availability_slots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    associate_id UUID NOT NULL REFERENCES associates(id) ON DELETE CASCADE,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    is_available BOOLEAN NOT NULL DEFAULT TRUE,
    slot_type TEXT NOT NULL DEFAULT 'standard' CHECK (slot_type IN ('standard', 'blocked', 'break', 'meeting')),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for availability_slots table
CREATE INDEX IF NOT EXISTS idx_availability_slots_associate_id ON availability_slots(associate_id);
CREATE INDEX IF NOT EXISTS idx_availability_slots_start_time ON availability_slots(start_time);
CREATE INDEX IF NOT EXISTS idx_availability_slots_available ON availability_slots(is_available);

-- Create the appointment_reminders table for tracking reminder notifications
CREATE TABLE IF NOT EXISTS appointment_reminders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    appointment_id UUID NOT NULL REFERENCES appointments(id) ON DELETE CASCADE,
    reminder_type TEXT NOT NULL CHECK (reminder_type IN ('email', 'sms', 'push')),
    reminder_time TIMESTAMP WITH TIME ZONE NOT NULL,
    sent_at TIMESTAMP WITH TIME ZONE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for appointment_reminders table
CREATE INDEX IF NOT EXISTS idx_appointment_reminders_appointment_id ON appointment_reminders(appointment_id);
CREATE INDEX IF NOT EXISTS idx_appointment_reminders_reminder_time ON appointment_reminders(reminder_time);
CREATE INDEX IF NOT EXISTS idx_appointment_reminders_status ON appointment_reminders(status);

-- Create triggers to automatically update the updated_at columns
CREATE TRIGGER trigger_update_associates_updated_at
    BEFORE UPDATE ON associates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_update_appointments_updated_at
    BEFORE UPDATE ON appointments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a view for appointment details with associate information
CREATE OR REPLACE VIEW appointment_details AS
SELECT 
    a.id,
    a.session_id,
    a.associate_id,
    ass.name as associate_name,
    ass.email as associate_email,
    ass.phone as associate_phone,
    ass.specialization as associate_specialization,
    a.user_name,
    a.user_email,
    a.user_phone,
    a.scheduled_time,
    a.appointment_type,
    a.status,
    a.notes,
    a.calendar_event_id,
    a.reminder_sent,
    a.created_at,
    a.updated_at
FROM appointments a
LEFT JOIN associates ass ON a.associate_id = ass.id;

-- Create a view for associate availability
CREATE OR REPLACE VIEW associate_availability AS
SELECT 
    ass.id as associate_id,
    ass.name as associate_name,
    ass.email as associate_email,
    ass.availability_hours,
    ass.timezone,
    COUNT(a.id) as upcoming_appointments,
    ass.is_active
FROM associates ass
LEFT JOIN appointments a ON ass.id = a.associate_id 
    AND a.scheduled_time > NOW() 
    AND a.status IN ('scheduled', 'confirmed')
WHERE ass.is_active = TRUE
GROUP BY ass.id, ass.name, ass.email, ass.availability_hours, ass.timezone, ass.is_active;

-- Function to get available time slots for an associate
CREATE OR REPLACE FUNCTION get_available_slots(
    p_associate_id UUID,
    p_start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    p_end_date TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '7 days',
    p_duration_minutes INTEGER DEFAULT 60
)
RETURNS TABLE (
    slot_start TIMESTAMP WITH TIME ZONE,
    slot_end TIMESTAMP WITH TIME ZONE,
    is_available BOOLEAN
) AS $$
DECLARE
    slot_time TIMESTAMP WITH TIME ZONE;
    slot_end_time TIMESTAMP WITH TIME ZONE;
    has_conflict BOOLEAN;
BEGIN
    -- Generate hourly slots between start and end dates
    slot_time := DATE_TRUNC('hour', p_start_date);
    
    WHILE slot_time <= p_end_date LOOP
        slot_end_time := slot_time + (p_duration_minutes || ' minutes')::INTERVAL;
        
        -- Check if this slot conflicts with existing appointments
        SELECT EXISTS(
            SELECT 1 FROM appointments 
            WHERE associate_id = p_associate_id 
            AND status IN ('scheduled', 'confirmed')
            AND (
                (scheduled_time <= slot_time AND scheduled_time + INTERVAL '1 hour' > slot_time)
                OR (scheduled_time < slot_end_time AND scheduled_time + INTERVAL '1 hour' >= slot_end_time)
                OR (scheduled_time >= slot_time AND scheduled_time + INTERVAL '1 hour' <= slot_end_time)
            )
        ) INTO has_conflict;
        
        -- Only return slots during business hours (9 AM - 6 PM, Monday-Friday)
        IF EXTRACT(DOW FROM slot_time) BETWEEN 1 AND 5 
           AND EXTRACT(HOUR FROM slot_time) BETWEEN 9 AND 17 THEN
            RETURN NEXT ROW(slot_time, slot_end_time, NOT has_conflict);
        END IF;
        
        slot_time := slot_time + INTERVAL '1 hour';
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to schedule an appointment
CREATE OR REPLACE FUNCTION schedule_appointment(
    p_session_id TEXT,
    p_associate_id UUID,
    p_user_name TEXT,
    p_user_email TEXT,
    p_user_phone TEXT,
    p_scheduled_time TIMESTAMP WITH TIME ZONE,
    p_appointment_type TEXT DEFAULT 'consultation',
    p_notes TEXT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    appointment_id UUID;
    has_conflict BOOLEAN;
BEGIN
    -- Check for scheduling conflicts
    SELECT EXISTS(
        SELECT 1 FROM appointments 
        WHERE associate_id = p_associate_id 
        AND status IN ('scheduled', 'confirmed')
        AND ABS(EXTRACT(EPOCH FROM (scheduled_time - p_scheduled_time))) < 3600
    ) INTO has_conflict;
    
    IF has_conflict THEN
        RAISE EXCEPTION 'Time slot conflict detected';
    END IF;
    
    -- Create the appointment
    INSERT INTO appointments (
        session_id, associate_id, user_name, user_email, user_phone,
        scheduled_time, appointment_type, notes
    ) VALUES (
        p_session_id, p_associate_id, p_user_name, p_user_email, p_user_phone,
        p_scheduled_time, p_appointment_type, p_notes
    ) RETURNING id INTO appointment_id;
    
    -- Create reminder records
    INSERT INTO appointment_reminders (appointment_id, reminder_type, reminder_time)
    VALUES 
        (appointment_id, 'email', p_scheduled_time - INTERVAL '24 hours'),
        (appointment_id, 'email', p_scheduled_time - INTERVAL '30 minutes');
    
    RETURN appointment_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get associate appointments
CREATE OR REPLACE FUNCTION get_associate_appointments(
    p_associate_id UUID,
    p_start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    p_end_date TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '30 days'
)
RETURNS TABLE (
    appointment_id UUID,
    session_id TEXT,
    user_name TEXT,
    user_email TEXT,
    scheduled_time TIMESTAMP WITH TIME ZONE,
    appointment_type TEXT,
    status TEXT,
    notes TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        a.id,
        a.session_id,
        a.user_name,
        a.user_email,
        a.scheduled_time,
        a.appointment_type,
        a.status,
        a.notes
    FROM appointments a
    WHERE a.associate_id = p_associate_id
    AND a.scheduled_time BETWEEN p_start_date AND p_end_date
    ORDER BY a.scheduled_time;
END;
$$ LANGUAGE plpgsql;

-- Function to get unique associates (for extracting from property data)
CREATE OR REPLACE FUNCTION get_unique_associates_from_properties()
RETURNS TABLE (
    name TEXT,
    email TEXT,
    specialization TEXT
) AS $$
BEGIN
    -- This would typically extract associates from property data
    -- For now, return sample data
    RETURN QUERY
    SELECT 
        'Sarah Johnson'::TEXT as name,
        'sarah.johnson@example.com'::TEXT as email,
        'Commercial Real Estate'::TEXT as specialization
    UNION ALL
    SELECT 
        'Michael Chen'::TEXT as name,
        'michael.chen@example.com'::TEXT as email,
        'Residential Real Estate'::TEXT as specialization;
END;
$$ LANGUAGE plpgsql;

-- Function to cancel an appointment
CREATE OR REPLACE FUNCTION cancel_appointment(p_appointment_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    affected_rows INTEGER;
BEGIN
    UPDATE appointments 
    SET status = 'cancelled', updated_at = NOW()
    WHERE id = p_appointment_id 
    AND status IN ('scheduled', 'confirmed');
    
    GET DIAGNOSTICS affected_rows = ROW_COUNT;
    RETURN affected_rows > 0;
END;
$$ LANGUAGE plpgsql;

-- Function to reschedule an appointment
CREATE OR REPLACE FUNCTION reschedule_appointment(
    p_appointment_id UUID,
    p_new_time TIMESTAMP WITH TIME ZONE
)
RETURNS BOOLEAN AS $$
DECLARE
    affected_rows INTEGER;
    p_associate_id UUID;
    has_conflict BOOLEAN;
BEGIN
    -- Get associate ID
    SELECT associate_id INTO p_associate_id
    FROM appointments
    WHERE id = p_appointment_id;
    
    -- Check for conflicts
    SELECT EXISTS(
        SELECT 1 FROM appointments 
        WHERE associate_id = p_associate_id 
        AND id != p_appointment_id
        AND status IN ('scheduled', 'confirmed')
        AND ABS(EXTRACT(EPOCH FROM (scheduled_time - p_new_time))) < 3600
    ) INTO has_conflict;
    
    IF has_conflict THEN
        RAISE EXCEPTION 'Time slot conflict detected';
    END IF;
    
    -- Update the appointment
    UPDATE appointments 
    SET scheduled_time = p_new_time, updated_at = NOW()
    WHERE id = p_appointment_id;
    
    -- Update reminder times
    UPDATE appointment_reminders 
    SET reminder_time = p_new_time - INTERVAL '24 hours'
    WHERE appointment_id = p_appointment_id AND reminder_type = 'email'
    AND reminder_time = (SELECT scheduled_time - INTERVAL '24 hours' FROM appointments WHERE id = p_appointment_id);
    
    GET DIAGNOSTICS affected_rows = ROW_COUNT;
    RETURN affected_rows > 0;
END;
$$ LANGUAGE plpgsql;

-- Insert sample associates
INSERT INTO associates (name, email, phone, specialization, availability_hours) VALUES
('Elizabeth Swann', 'Elizabeth@example.com', '(555) 123-4567', 'Commercial Real Estate', '9:00 AM - 6:00 PM'),
('Cutler Beckett', 'Cutler@example.com', '(555) 234-5678', 'Residential Real Estate', '8:00 AM - 7:00 PM'),
('Jack Sparrow', 'Jack@example.com', '(555) 345-6789', 'Property Management', '9:00 AM - 5:00 PM'),
('Davy Jones', 'Davy@example.com', '(555) 456-7890', 'Investment Properties', '10:00 AM - 6:00 PM'),
('Hector Barbossa', 'Hector@example.com', '(555) 456-7890', 'Investment Properties', '10:00 AM - 6:00 PM')
ON CONFLICT (email) DO NOTHING;



-- Comments for setup instructions
/*
SETUP INSTRUCTIONS:

1. Run this script after setting up the main database schema
2. This creates the appointment scheduling tables and functions
3. The script includes sample associates data
4. Make sure the update_updated_at_column function exists from the main schema
5. Test the functions with sample data to ensure they work correctly

IMPORTANT NOTES:
- All appointment times are stored with timezone information
- The system assumes business hours are 9 AM - 6 PM, Monday-Friday
- Appointment conflicts are checked within 1-hour windows
- Reminders are automatically created for 24 hours and 30 minutes before appointments
- The system supports multiple reminder types (email, SMS, push)
*/ 