from __future__ import annotations
from ..components.acceptance import MiCROAcceptancePolicyMulti
from ..components.offering import MiCROOfferingPolicyMulti
from .modular.mapneg import MAPNegotiator

__all__ = ["MiCRONegotiatorMulti"]

from ... import MiCROAcceptancePolicyMulti


class MiCRONegotiatorMulti(MAPNegotiator):
    """
    Rational Concession Negotiator

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides preferences)
         owner: The Agent that owns the negotiator.

    """

    def __init__(self, *args, accept_same: bool = True, **kwargs):
        kwargs["offering"] = MiCROOfferingPolicyMulti()
        kwargs["acceptance"] = MiCROAcceptancePolicyMulti(kwargs["offering"], accept_same)
        super().__init__(*args, **kwargs)
