from __future__ import annotations


def scenario_message_class():
    """Return a wire-compatible minimal WOMD ``Scenario`` protobuf class.

    The full Waymo wheel is tied to specific TensorFlow/Python releases.  The
    paper pipeline only needs motion fields, so this descriptor intentionally
    declares only their original field numbers.  Protobuf safely ignores the
    map, lidar, camera, and traffic-light fields that are not declared here.
    """

    try:
        from google.protobuf import descriptor_pb2, descriptor_pool
        from google.protobuf.message_factory import GetMessageClass
    except ImportError as exc:  # pragma: no cover - optional data dependency
        raise ImportError(
            "Official WOMD loading requires protobuf: pip install 'protobuf>=5'"
        ) from exc

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "predictive_pc_fmcw/womd_minimal_scenario.proto"
    file_proto.package = "waymo.open_dataset"
    file_proto.syntax = "proto2"

    state = file_proto.message_type.add()
    state.name = "ObjectState"
    for name, number, field_type in (
        ("center_x", 2, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE),
        ("center_y", 3, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE),
        ("valid", 11, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
    ):
        field = state.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = field_type

    track = file_proto.message_type.add()
    track.name = "Track"
    for name, number in (("id", 1), ("object_type", 2)):
        field = track.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT32
    states = track.field.add()
    states.name = "states"
    states.number = 3
    states.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    states.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    states.type_name = ".waymo.open_dataset.ObjectState"

    scenario = file_proto.message_type.add()
    scenario.name = "Scenario"
    timestamps = scenario.field.add()
    timestamps.name = "timestamps_seconds"
    timestamps.number = 1
    timestamps.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    timestamps.type = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
    tracks = scenario.field.add()
    tracks.name = "tracks"
    tracks.number = 2
    tracks.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    tracks.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    tracks.type_name = ".waymo.open_dataset.Track"
    for name, number, field_type in (
        ("scenario_id", 5, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
        ("sdc_track_index", 6, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
        ("current_time_index", 10, descriptor_pb2.FieldDescriptorProto.TYPE_INT32),
    ):
        field = scenario.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = field_type

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    descriptor = pool.FindMessageTypeByName("waymo.open_dataset.Scenario")
    return GetMessageClass(descriptor)
