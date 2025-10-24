'''Contains methods and classes common for the whole source code.'''
import pydantic


model_config = pydantic.ConfigDict(
    extra='allow',
    strict=True,
    arbitrary_types_allowed=True
)

custom_validate_call = pydantic.validate_call(
    config=dict(
        strict=True,
        validate_return=True,
        arbitrary_types_allowed=True
    )
)
