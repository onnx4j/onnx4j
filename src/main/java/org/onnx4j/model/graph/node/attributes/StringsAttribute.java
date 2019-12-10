package org.onnx4j.model.graph.node.attributes;

import java.util.List;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.onnx.prototypes.OnnxProto3.AttributeProto;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.google.protobuf.ByteString;

public class StringsAttribute extends Attribute<List<String>> {

	public <T> StringsAttribute(AttributeProto attrProto) {
		super(Lists.transform(attrProto.getStringsList(), new Function<ByteString, String>() {

			@Override
			public @Nullable String apply(@Nullable ByteString input) {
				// TODO Auto-generated method stub
				return input.toStringUtf8();
			}

		}), attrProto.getName(), attrProto.getDocString());
	}

	@Override
	public void close() throws Exception {}

}
