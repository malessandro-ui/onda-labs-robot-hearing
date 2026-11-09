"""Demonstrate policy feature encoding without requiring a simulator."""
from onda_hearing.policy import AcousticEvidence, HearingObservationBank, POLICY_FEATURES
import math

bank = HearingObservationBank(2)
bank.ingest(0, AcousticEvidence(0, math.radians(38), .48, 'world', True))
print('features:', POLICY_FEATURES)
print('source +38, other silent:', bank.read(100_000_000))
bank.reset([0])
print('after reset:', bank.read(0))
