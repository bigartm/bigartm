"""
Exceptions corresponding to BigARTM error codes
"""

ARTM_SUCCESS = 0
ARTM_STILL_WORKING = -1

# TODO: add docstrings


class ArtmException(BaseException):
    pass


class InternalError(ArtmException):
    pass


class ArgumentOutOfRangeException(ArtmException):
    pass


class InvalidMasterIdException(ArtmException):
    pass


class CorruptedMessageException(ArtmException):
    pass


class InvalidOperationException(ArtmException):
    pass


class DiskReadException(ArtmException):
    pass


class DiskWriteException(ArtmException):
    pass


ARTM_EXCEPTION_BY_CODE = {
    -2: InternalError,
    -3: ArgumentOutOfRangeException,
    -4: InvalidMasterIdException,
    -5: CorruptedMessageException,
    -6: InvalidOperationException,
    -7: DiskReadException,
    -8: DiskWriteException,
}
