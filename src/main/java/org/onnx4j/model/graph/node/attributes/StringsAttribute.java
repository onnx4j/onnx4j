/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.model.graph.node.attributes;

import java.util.Collections;
import java.util.List;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.onnx4j.model.graph.node.Attribute;
import org.onnx4j.prototypes.OnnxProto3.AttributeProto;

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

	/**
	 * 由于是引用传递，这里返回一个不可修改的List对象，防止Operator在执行的过程中修改List对象的值。
	 */
	@Override
	public List<String> getValue() {
		return Collections.unmodifiableList(super.getValue());
	}

}