CREATE TABLE IF NOT EXISTS property_engagements (
    id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    property_id int NOT NULL,               -- unique_id from HackathonInternalKnowledgeBase.csv
    event_type text NOT NULL CHECK (event_type IN ('view','click','mention')),
    session_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS property_engagements_property_id_idx ON property_engagements(property_id);
CREATE INDEX IF NOT EXISTS property_engagements_created_at_idx ON property_engagements(created_at); 